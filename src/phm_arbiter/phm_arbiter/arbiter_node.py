"""ROS 2 arbiter node: fuses DetectorVerdict -> /phm/health (PolicyHealthStatus).

This module is a thin ROS 2 adapter over the pure-Python arbitration logic in
``_core.py``. All fusion logic lives in ``_core.arbitrate()`` and is tested
without a ROS graph.

Single-writer invariant: exactly one instance of this node publishes
``/phm/health`` (mirrors HELIX safety_envelope_node single-publisher contract,
ported from helix_recovery/recovery_node.py:142 in the HELIX repo).

Spec reference: docs/superpowers/specs/2026-05-29-policy-health-monitor-design.md
section 3.3.

Topics:
  Subscribed:  /phm/verdicts  (phm_msgs/DetectorVerdict)
               QoS: reliable, keep_last 10
  Published:   /phm/health    (phm_msgs/PolicyHealthStatus)
               QoS: reliable, keep_last 1, transient_local (latecomer gets state)

Parameters (loaded from config/phm_arbiter.yaml):
  staleness_sec  float  1.0   Age (seconds) beyond which a verdict is stale.
  timer_period   float  0.05  Arbitration period in seconds (20 Hz default).
"""

from __future__ import annotations

import math
import time
from typing import Any

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)

from phm_arbiter._core import (
    STATE_OK,
    PolicyHealthStatusData,
    arbitrate,
)


class ArbiterNode(Node):
    """Fuses DetectorVerdict messages into a single /phm/health output."""

    def __init__(self) -> None:
        super().__init__("phm_arbiter")

        # ---------- Parameters ----------
        self.declare_parameter(
            "staleness_sec",
            1.0,
        )
        self.declare_parameter(
            "timer_period",
            0.05,
        )

        staleness_sec: float = (
            self.get_parameter("staleness_sec").get_parameter_value().double_value
        )
        timer_period: float = (
            self.get_parameter("timer_period").get_parameter_value().double_value
        )

        # ---------- State ----------
        # Latest verdict per source. Key = source string.
        self._latest: dict[str, Any] = {}

        # ---------- QoS ----------
        # All four policies declared explicitly on every PHM endpoint so intent
        # is unambiguous and a future rclpy default change cannot silently alter
        # behavior (review decision 7).
        # Verdicts: reliable, keep_last 10, volatile (commands channel, must not drop).
        verdict_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        # Health output: reliable, keep_last 1, transient_local so late-joining
        # subscribers (e.g. a recovery node that starts after the arbiter) receive
        # the most recent health state immediately. The recovery subscriber must
        # also declare TRANSIENT_LOCAL or DDS drops every message (review decision 7).
        health_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        # ---------- Interfaces ----------
        # Deferred imports of phm_msgs to keep this import-safe in colcon build
        # environments where the overlay may not be sourced at import time.
        from phm_msgs.msg import DetectorVerdict, PolicyHealthStatus  # noqa: PLC0415

        self._PolicyHealthStatus = PolicyHealthStatus

        self._sub = self.create_subscription(
            DetectorVerdict,
            "/phm/verdicts",
            self._verdict_callback,
            verdict_qos,
        )

        self._pub = self.create_publisher(
            PolicyHealthStatus,
            "/phm/health",
            health_qos,
        )

        self._staleness_sec = staleness_sec
        self._timer = self.create_timer(timer_period, self._timer_callback)

        self.get_logger().info(
            f"phm_arbiter started (staleness={staleness_sec}s, period={timer_period}s)"
        )

    def _verdict_callback(self, msg: Any) -> None:  # msg: DetectorVerdict
        """Store the latest verdict per source (last-write-wins per source key)."""
        # Attach a wall-clock receive timestamp so the arbiter can detect
        # staleness even when the message header stamp is zero.
        msg._recv_time = time.monotonic()
        self._latest[msg.source] = msg

    def _timer_callback(self) -> None:
        """Run arbitration and publish the fused health status."""
        now = time.monotonic()

        # Build a list of lightweight wrappers so _core.arbitrate() sees plain
        # Python attributes (source, score, violating, reason, suggested_action,
        # timestamp). We use _recv_time as the timestamp because the detectors
        # may not stamp their headers.
        class _MsgView:
            __slots__ = (
                "source",
                "score",
                "violating",
                "reason",
                "suggested_action",
                "timestamp",
            )

        views = []
        for msg in self._latest.values():
            v = _MsgView()
            v.source = msg.source
            # Trust-boundary sanitize (review decision 6): never forward a
            # non-finite score from a misbehaving/poisoned detector into the
            # fusion. _core.arbitrate() also guards this (defense in depth), but
            # we sanitize at ingestion and log throttled so an operator sees it.
            raw_score = float(msg.score)
            if not math.isfinite(raw_score):
                self.get_logger().warning(
                    f"non-finite score {raw_score!r} from source "
                    f"{msg.source!r}; treating detector as DEGRADED "
                    "(bad-score sentinel)",
                    throttle_duration_sec=1.0,
                )
            v.score = raw_score  # _core sanitizes; pass through so reason is set there
            v.violating = bool(msg.violating)
            v.reason = msg.reason
            v.suggested_action = int(msg.suggested_action)
            v.timestamp = getattr(msg, "_recv_time", now)
            views.append(v)

        result: PolicyHealthStatusData = arbitrate(views, now, self._staleness_sec)

        # Final belt-and-suspenders: never publish a NaN/inf in PolicyHealthStatus.score.
        if not math.isfinite(result.score):
            result = PolicyHealthStatusData(
                state=result.state,
                score=0.5,
                reason=result.reason,
                source=result.source,
                suggested_action=result.suggested_action,
            )

        out = self._PolicyHealthStatus()
        out.header.stamp = self.get_clock().now().to_msg()
        out.state = result.state
        out.score = float(result.score)
        out.reason = result.reason
        out.source = result.source
        out.suggested_action = result.suggested_action

        self._pub.publish(out)

        if result.state != STATE_OK:
            self.get_logger().info(
                f"health={result.state} score={result.score:.3f} "
                f"source={result.source!r} reason={result.reason!r}",
                throttle_duration_sec=1.0,
            )


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = ArbiterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
