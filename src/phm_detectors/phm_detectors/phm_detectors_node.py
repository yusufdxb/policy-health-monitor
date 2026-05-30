"""rclpy node: runs phm_detectors adapters off the live ROS 2 graph and
publishes ``phm_msgs/DetectorVerdict`` to ``/phm/verdicts``.

This node is a THIN WRAPPER over the pure-Python adapters in ``_core.py``. It
owns the ROS interface (subscriptions, timers, publishers) but contains no
detection logic. All detection logic lives in the adapter classes so it can be
unit-tested without a ROS graph.

Topics published:
  /phm/verdicts  (phm_msgs/DetectorVerdict)  QoS: reliable, keep_last 10.

Parameters (declared from ``config/detectors.yaml``):
  freq_topics              string[]  topic names to monitor for frequency drops
  freq_tolerance_percent   float     drop below this % of baseline triggers (default 20.0)
  freq_min_consecutive     int       hysteresis window for freq (default 2)
  threshold_cpu_limit      float     cpu_percent upper bound (default 80.0)
  threshold_mem_limit      float     memory_percent upper bound (default 85.0)
  threshold_temp_limit     float     gpu_temp_c upper bound (default 85.0)
  threshold_min_consecutive int      hysteresis window for threshold (default 2)
  dead_timeout_sec         float     silence window for dead-topic detection (default 5.0)
  dead_topics              string[]  topic names to monitor for dead-topic

The node uses a 1 Hz graph-stats timer to drive dead-topic and threshold
adapters. Frequency adapters are fed from a separate per-topic subscription
(topic_stats) if available, otherwise from message callbacks.
"""

from __future__ import annotations

import time
from typing import Any

import rclpy
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)

from phm_detectors._core import (
    DeadTopicAdapter,
    DeadTopicSample,
    FrequencyDropAdapter,
    FrequencySample,
    StaticThresholdAdapter,
)

# phm_msgs is only available after colcon build; import guarded for pure-python
# tests (tests target _core.py directly and never import this file).
try:
    from phm_msgs.msg import DetectorVerdict
    _HAS_PHM_MSGS = True
except ImportError:  # not built yet; node cannot run but core tests still pass
    _HAS_PHM_MSGS = False


# QoS for /phm/verdicts (review decision 7): all four policies explicit.
# Verdicts are control-critical; best_effort risks dropping a STOP, so RELIABLE.
# Live stream -> VOLATILE durability (matches the arbiter's verdict subscriber).
_VERDICT_QOS = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.VOLATILE,
)

# QoS for the per-topic liveness/frequency subscriptions. We only need arrival
# timing, so BEST_EFFORT is acceptable and matches typical sensor publishers,
# but all four policies are declared explicitly (review decision 7).
_WATCH_QOS = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10,
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE,
)


class PhmDetectorsNode(Node):
    """Runs frequency / threshold / dead-topic adapters and publishes verdicts."""

    def __init__(self) -> None:
        super().__init__("phm_detectors")

        # -- Declare parameters (spec §3.2: declare_parameter + descriptor) --
        self._declare_params()

        freq_topics: list[str] = self._str_array_param("freq_topics")
        freq_tol: float = (
            self.get_parameter("freq_tolerance_percent")
            .get_parameter_value()
            .double_value
        )
        freq_min_consec: int = (
            self.get_parameter("freq_min_consecutive")
            .get_parameter_value()
            .integer_value
        )
        cpu_lim: float = (
            self.get_parameter("threshold_cpu_limit")
            .get_parameter_value()
            .double_value
        )
        mem_lim: float = (
            self.get_parameter("threshold_mem_limit")
            .get_parameter_value()
            .double_value
        )
        temp_lim: float = (
            self.get_parameter("threshold_temp_limit")
            .get_parameter_value()
            .double_value
        )
        thresh_min_consec: int = (
            self.get_parameter("threshold_min_consecutive")
            .get_parameter_value()
            .integer_value
        )
        dead_timeout: float = (
            self.get_parameter("dead_timeout_sec")
            .get_parameter_value()
            .double_value
        )
        dead_topics: list[str] = self._str_array_param("dead_topics")

        # -- Build adapters ---------------------------------------------------
        self._freq_adapters: dict[str, FrequencyDropAdapter] = {
            t: FrequencyDropAdapter(t, freq_tol, freq_min_consec) for t in freq_topics
        }
        self._thresh_adapters: dict[str, StaticThresholdAdapter] = {
            "cpu_percent": StaticThresholdAdapter(
                "system:cpu", "cpu_percent", thresh_min_consec
            ),
            "memory_percent": StaticThresholdAdapter(
                "system:mem", "memory_percent", thresh_min_consec
            ),
            "gpu_temp_c": StaticThresholdAdapter(
                "system:gpu", "gpu_temp_c", thresh_min_consec
            ),
        }
        self._thresh_limits: dict[str, float] = {
            "cpu_percent": cpu_lim,
            "memory_percent": mem_lim,
            "gpu_temp_c": temp_lim,
        }
        self._dead_adapters: dict[str, DeadTopicAdapter] = {
            t: DeadTopicAdapter(t, dead_timeout) for t in dead_topics
        }
        # last-seen clock for dead-topic adapters, keyed by topic.
        self._last_seen: dict[str, float] = {
            t: self._now() for t in dead_topics
        }

        # -- Publisher --------------------------------------------------------
        if _HAS_PHM_MSGS:
            self._verdict_pub = self.create_publisher(
                DetectorVerdict, "/phm/verdicts", _VERDICT_QOS
            )

        # -- Per-topic liveness/frequency subscriptions -----------------------
        # The node subscribes to each monitored topic with a generic callback
        # that (a) feeds a FrequencyDropAdapter with a synthetic Hz estimate
        # and (b) refreshes the dead-topic last_seen timestamp.
        #
        # Humble's rclpy does not expose a usable generic AnyMsg, so the message
        # type is resolved from the ROS graph (get_topic_names_and_types) and
        # imported dynamically (rosidl_runtime_py.utilities.get_message). A topic
        # whose type is not yet on the graph cannot be subscribed at construction;
        # a retry timer re-attempts so the subscription binds once the publisher
        # appears. The node MUST construct without crashing in either case
        # (review decision 8).
        self._msg_counts: dict[str, int] = {}
        self._window_start: dict[str, float] = {}
        self._watch_subs: dict[str, Any] = {}
        self._watch_topics: list[str] = sorted(set(freq_topics) | set(dead_topics))

        self._try_bind_watch_subs()
        if len(self._watch_subs) < len(self._watch_topics):
            # Some topic types were not resolvable yet; retry on a slow timer.
            self._resolve_timer = self.create_timer(2.0, self._try_bind_watch_subs)

        # 1 Hz timer drives dead-topic checks and (future) system metrics.
        self._timer = self.create_timer(1.0, self._tick)

        self.get_logger().info(
            "phm_detectors_node ready: "
            f"{len(self._freq_adapters)} freq, "
            f"{len(self._thresh_adapters)} threshold, "
            f"{len(self._dead_adapters)} dead-topic adapters"
        )

    # ------------------------------------------------------------------
    # Parameter declaration
    # ------------------------------------------------------------------

    def _declare_params(self) -> None:
        """Declare all node parameters with descriptors."""
        self.declare_parameter(
            "freq_topics",
            None,
            ParameterDescriptor(
                # Declare the type explicitly so a configured STRING_ARRAY
                # override is accepted. An empty-list default infers BYTE_ARRAY
                # and rejects a string-array override (the node could never
                # receive configured topics otherwise). The default is None so
                # the descriptor type is authoritative; an unset param reads as
                # an empty list via get_parameter_or (see _str_array_param).
                type=ParameterType.PARAMETER_STRING_ARRAY,
                description="Topic names to monitor for frequency drops (string array)",
            ),
        )
        self.declare_parameter(
            "freq_tolerance_percent",
            20.0,
            ParameterDescriptor(
                description=(
                    "Frequency must stay above baseline*(1-tol/100) to be healthy"
                )
            ),
        )
        self.declare_parameter(
            "freq_min_consecutive",
            2,
            ParameterDescriptor(
                description="Hysteresis window: consecutive violations before freq fires"
            ),
        )
        self.declare_parameter(
            "threshold_cpu_limit",
            80.0,
            ParameterDescriptor(description="cpu_percent upper bound (default 80%)"),
        )
        self.declare_parameter(
            "threshold_mem_limit",
            85.0,
            ParameterDescriptor(description="memory_percent upper bound (default 85%)"),
        )
        self.declare_parameter(
            "threshold_temp_limit",
            85.0,
            ParameterDescriptor(description="gpu_temp_c upper bound (default 85 C)"),
        )
        self.declare_parameter(
            "threshold_min_consecutive",
            2,
            ParameterDescriptor(
                description="Hysteresis window: consecutive violations before threshold fires"
            ),
        )
        self.declare_parameter(
            "dead_timeout_sec",
            5.0,
            ParameterDescriptor(description="Silence after which a topic is dead (sec)"),
        )
        self.declare_parameter(
            "dead_topics",
            None,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING_ARRAY,
                description="Topic names to monitor for dead-topic (string array)",
            ),
        )

    # ------------------------------------------------------------------
    # Subscription binding (graph type resolution)
    # ------------------------------------------------------------------

    def _try_bind_watch_subs(self) -> None:
        """Bind a liveness/frequency subscription for each watched topic whose
        type is resolvable on the ROS graph.

        Resolves the topic's message type via the graph and imports it with
        rosidl_runtime_py.utilities.get_message, then subscribes with an explicit
        QoSProfile. Topics not yet on the graph are skipped and retried by the
        resolve timer. Never raises (review decision 8): a topic that cannot be
        resolved leaves the node running on the 1 Hz timer only.
        """
        try:
            graph = dict(self.get_topic_names_and_types())
        except Exception as exc:  # graph query failed; retry next tick
            self.get_logger().info(
                f"topic graph query failed ({exc}); will retry",
                throttle_duration_sec=5.0,
            )
            return

        for topic in self._watch_topics:
            if topic in self._watch_subs:
                continue  # already bound
            type_names = graph.get(topic)
            if not type_names:
                continue  # not on the graph yet; retry later
            try:
                from rosidl_runtime_py.utilities import get_message
                msg_type = get_message(type_names[0])
                sub = self.create_subscription(
                    msg_type,
                    topic,
                    lambda msg, t=topic: self._on_msg(t, msg),
                    _WATCH_QOS,
                )
                self._watch_subs[topic] = sub
                self.get_logger().info(
                    f"bound liveness/frequency subscription on {topic} "
                    f"({type_names[0]})"
                )
            except Exception as exc:
                self.get_logger().info(
                    f"could not subscribe to {topic} ({type_names[0]}): {exc}; "
                    "will retry",
                    throttle_duration_sec=5.0,
                )

        # Stop the resolve timer once every watched topic is bound.
        if (
            getattr(self, "_resolve_timer", None) is not None
            and len(self._watch_subs) >= len(self._watch_topics)
        ):
            self._resolve_timer.cancel()

    # ------------------------------------------------------------------
    # ROS callbacks
    # ------------------------------------------------------------------

    def _on_msg(self, topic: str, _msg: Any) -> None:
        """Record a message arrival and refresh frequency / dead-topic state."""
        now = self._now()

        # Refresh dead-topic liveness.
        if topic in self._dead_adapters:
            self._dead_adapters[topic].mark_alive(now)
            self._last_seen[topic] = now

        # Count messages for rolling frequency estimate.
        if topic in self._freq_adapters:
            if topic not in self._msg_counts:
                self._msg_counts[topic] = 0
                self._window_start[topic] = now
            self._msg_counts[topic] += 1

    def _tick(self) -> None:
        """1 Hz timer: drive dead-topic adapters and emit verdicts."""
        now = self._now()

        # Dead-topic verdicts.
        for topic, adapter in self._dead_adapters.items():
            last_seen = self._last_seen.get(topic, now)
            sample = DeadTopicSample(topic=topic, last_seen_sec=last_seen, now_sec=now)
            verdict = adapter.update(sample)
            if verdict is not None:
                self._publish(verdict)

        # Frequency verdicts: derive Hz from message counts over the elapsed window.
        for topic, adapter in self._freq_adapters.items():
            count = self._msg_counts.get(topic, 0)
            t0 = self._window_start.get(topic, now)
            elapsed = now - t0
            hz = count / elapsed if elapsed > 0 else 0.0

            sample = FrequencySample(topic=topic, frequency_hz=hz)
            verdict = adapter.update(sample)
            if verdict is not None:
                self._publish(verdict)

            # Reset window each tick for a 1-second rolling average.
            self._msg_counts[topic] = 0
            self._window_start[topic] = now

        # Throttled log so we confirm the timer is running.
        self.get_logger().info(
            "phm_detectors tick",
            throttle_duration_sec=5.0,
        )

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    def _publish(self, verdict: Any) -> None:
        """Stamp and publish a DetectorVerdictData as a ROS DetectorVerdict."""
        if not _HAS_PHM_MSGS:
            return
        msg = DetectorVerdict()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.source = verdict.source
        msg.score = float(verdict.score)
        msg.violating = bool(verdict.violating)
        msg.reason = verdict.reason
        msg.suggested_action = int(verdict.suggested_action)
        self._verdict_pub.publish(msg)

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _str_array_param(self, name: str) -> list[str]:
        """Read a STRING_ARRAY parameter, returning [] if it was never set.

        The array params are declared with a None default and an explicit
        STRING_ARRAY descriptor type so a configured override is accepted (an
        empty-list default would infer BYTE_ARRAY and reject the override). When
        no override is provided the param is NOT_SET, so get_parameter_or yields
        an empty string array rather than raising.
        """
        param = self.get_parameter_or(name)
        return list(param.get_parameter_value().string_array_value)

    @staticmethod
    def _now() -> float:
        """Wall-clock seconds (float). Used for dead-topic elapsed math."""
        return time.monotonic()


def main(args: list[str] | None = None) -> None:
    """Entry point for ``ros2 run phm_detectors phm_detectors_node``."""
    rclpy.init(args=args)
    node = PhmDetectorsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
