"""phm_ood lifecycle node: PolicyEmbedding -> DetectorVerdict.

This node subscribes to PolicyEmbedding messages on a configurable topic,
feeds each embedding into the pure-Python OodCore, and publishes a
DetectorVerdict on /phm/verdicts. It is a thin ROS adapter; all decision
logic lives in phm_ood._core.OodCore so it can be tested without a ROS graph.

QoS choices (per spec section 2 conventions):
  - Subscription (/policy/embedding): best_effort, keep_last(10).
    Policy embedding streams are sensor-like: dropping a frame is preferable
    to blocking on a reliable backlog that adds latency.
  - Publisher (/phm/verdicts): reliable, keep_last(10).
    Verdicts drive downstream safety decisions; reliable ensures the arbiter
    receives them.

Parameters (all in config/phm_ood.yaml):
  embedding_topic     : topic to subscribe to (default /policy/embedding)
  window              : rolling-spread window length (default 30)
  threshold           : pre-calibrated spread threshold (default 0.0)
  calibration_file    : optional path to a .npz with key 'threshold' (default "")
  hysteresis_count    : consecutive violating frames before firing (default 3)
  compute_every       : spread computation frequency gate (default 1)
  embed_dim           : expected embedding dim for validation, 0=skip (default 0)
"""

from __future__ import annotations

import numpy as np
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)

from phm_ood._core import OodCore

# Deferred ROS message imports (avoid import at module level in no-ROS tests).
try:
    from phm_msgs.msg import DetectorVerdict, PolicyEmbedding  # type: ignore[import]
except ImportError:
    DetectorVerdict = None  # type: ignore[assignment,misc]
    PolicyEmbedding = None  # type: ignore[assignment,misc]

# QoS: subscriber (best_effort, depth 10) -- sensor-like stream.
_SUB_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    durability=DurabilityPolicy.VOLATILE,
)

# QoS: publisher (reliable, depth 10) -- safety-critical verdict.
_PUB_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    durability=DurabilityPolicy.VOLATILE,
)

_VERDICT_TOPIC = "/phm/verdicts"


def _pd(desc: str, read_only: bool = False) -> ParameterDescriptor:
    """Build a ParameterDescriptor with description and optional read_only."""
    return ParameterDescriptor(description=desc, read_only=read_only)


class OodNode(LifecycleNode):
    """Lifecycle node that runs the rolling-spread OOD detector.

    Lifecycle:
      configure  : declare parameters, instantiate OodCore, load calibration.
      activate   : create subscriber + publisher.
      deactivate : destroy subscriber + publisher, keep core state.
      cleanup    : reset core state.
      shutdown   : (handled by base class destruction).
    """

    def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
        super().__init__("phm_ood", **kwargs)
        self._core: OodCore | None = None
        self._sub = None
        self._pub = None

    # ------------------------------------------------------------------
    # Lifecycle callbacks
    # ------------------------------------------------------------------

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Declare parameters and build the OodCore."""
        self.declare_parameter(
            "embedding_topic",
            "/policy/embedding",
            _pd("Topic to subscribe to for PolicyEmbedding messages.",
                read_only=True),
        )
        self.declare_parameter(
            "window",
            30,
            _pd("Rolling-spread window length (number of frames). Must be >= 2."),
        )
        self.declare_parameter(
            "threshold",
            0.0,
            _pd(
                "Pre-calibrated spread threshold. spread < threshold flags OOD."
                " Overridden by calibration_file if provided."
            ),
        )
        self.declare_parameter(
            "calibration_file",
            "",
            _pd(
                "Path to a .npz file with key 'threshold'. If non-empty, loads"
                " threshold from file and ignores the 'threshold' parameter.",
                read_only=True,
            ),
        )
        self.declare_parameter(
            "hysteresis_count",
            3,
            _pd(
                "Consecutive violating frames required before the verdict flips"
                " to violating=True. Must be >= 1."
            ),
        )
        self.declare_parameter(
            "compute_every",
            1,
            _pd(
                "Frequency gate: compute spread every N frames, re-use last"
                " verdict otherwise. 1 = every frame."
            ),
        )
        self.declare_parameter(
            "embed_dim",
            0,
            _pd("Expected embedding dimension for input validation. 0 = skip."),
        )

        window = self.get_parameter("window").get_parameter_value().integer_value
        hysteresis_count = (
            self.get_parameter("hysteresis_count")
            .get_parameter_value()
            .integer_value
        )
        compute_every = (
            self.get_parameter("compute_every")
            .get_parameter_value()
            .integer_value
        )
        embed_dim = (
            self.get_parameter("embed_dim").get_parameter_value().integer_value
        )

        # Validate parameters before constructing core.
        if window < 2:
            self.get_logger().error(
                f"Parameter 'window' must be >= 2, got {window}. Failing configure."
            )
            return TransitionCallbackReturn.FAILURE

        # Resolve threshold.
        threshold = self.get_parameter("threshold").get_parameter_value().double_value
        calib_file = (
            self.get_parameter("calibration_file")
            .get_parameter_value()
            .string_value
        )
        if calib_file:
            try:
                data = np.load(calib_file)
                threshold = float(data["threshold"])
                self.get_logger().info(
                    f"Loaded threshold {threshold:.6f} from {calib_file}"
                )
            except Exception as exc:
                self.get_logger().error(
                    f"Failed to load calibration_file '{calib_file}': {exc}."
                    " Falling back to 'threshold' parameter."
                )

        self._core = OodCore(
            window=window,
            threshold=threshold,
            hysteresis_count=hysteresis_count,
            compute_every=compute_every,
            embed_dim=embed_dim,
        )

        self.get_logger().info(
            f"OodNode configured: window={window}, threshold={threshold:.6f},"
            f" hysteresis_count={hysteresis_count}, compute_every={compute_every}"
        )
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Create subscriber and publisher."""
        if self._core is None:
            self.get_logger().error("Core not initialized; call configure first.")
            return TransitionCallbackReturn.FAILURE

        if DetectorVerdict is None or PolicyEmbedding is None:
            self.get_logger().error(
                "phm_msgs not available; cannot create subscriber/publisher."
            )
            return TransitionCallbackReturn.FAILURE

        topic = (
            self.get_parameter("embedding_topic")
            .get_parameter_value()
            .string_value
        )

        self._sub = self.create_subscription(
            PolicyEmbedding,
            topic,
            self._embedding_callback,
            _SUB_QOS,
        )
        self._pub = self.create_lifecycle_publisher(
            DetectorVerdict,
            _VERDICT_TOPIC,
            _PUB_QOS,
        )

        self.get_logger().info(
            f"OodNode activated: subscribing {topic} -> publishing {_VERDICT_TOPIC}"
        )
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Destroy subscriber and publisher; keep core state."""
        if self._sub is not None:
            self.destroy_subscription(self._sub)
            self._sub = None
        if self._pub is not None:
            self.destroy_publisher(self._pub)
            self._pub = None
        self.get_logger().info("OodNode deactivated.")
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Reset core state (discard rolling buffer and hysteresis)."""
        self._core = None
        self.get_logger().info("OodNode cleaned up.")
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Shutdown: destroy remaining resources."""
        if self._sub is not None:
            self.destroy_subscription(self._sub)
            self._sub = None
        if self._pub is not None:
            self.destroy_publisher(self._pub)
            self._pub = None
        self.get_logger().info("OodNode shutting down.")
        return TransitionCallbackReturn.SUCCESS

    # ------------------------------------------------------------------
    # Subscription callback
    # ------------------------------------------------------------------

    def _embedding_callback(self, msg) -> None:  # type: ignore[no-untyped-def]
        """Receive a PolicyEmbedding, run OodCore.update, publish verdict."""
        if self._core is None:
            return

        # Validate dim field against the embedding length.
        embedding = np.asarray(msg.embedding, dtype=np.float32)
        if msg.dim > 0 and int(msg.dim) != len(embedding):
            self.get_logger().warn(
                f"PolicyEmbedding.dim={msg.dim} != len(embedding)={len(embedding)};"
                " trusting the array.",
                throttle_duration_sec=1.0,
            )

        verdict_data = self._core.update(embedding, policy_id=msg.policy_id)

        # Publish only when the lifecycle publisher is active.
        if self._pub is None or not self._pub.is_activated:
            return

        out = DetectorVerdict()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = ""
        out.source = verdict_data.source
        out.score = float(verdict_data.score)
        out.violating = bool(verdict_data.violating)
        out.reason = verdict_data.reason
        out.suggested_action = int(verdict_data.suggested_action)

        self._pub.publish(out)

        # Throttled log: only log every second to avoid per-tick spam.
        self.get_logger().info(
            f"OOD: spread={self._core.last_spread:.4f}"
            f" thr={self._core.threshold:.4f}"
            f" violating={verdict_data.violating}"
            f" score={verdict_data.score:.3f}",
            throttle_duration_sec=1.0,
        )
