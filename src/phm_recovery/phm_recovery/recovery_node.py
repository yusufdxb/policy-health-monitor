"""PHM Recovery Node.

Subscribes to /phm/health (phm_msgs/PolicyHealthStatus) and turns
INTERVENE / STOP into safe zero-velocity holds on /phm/cmd_vel
(geometry_msgs/Twist). The zero-velocity Twist is re-published at a
configurable rate while the hold is active.

Architecture:
- All policy decisions live in phm_recovery._core (pure Python, tested without
  a ROS graph): SafetyEnvelope, HealthToActionMapper, RewindHook.
- This node is a thin adapter: it owns the ROS subscriptions, publishers, timer,
  parameter declarations, and lifecycle callbacks. It delegates every decision.

The node is a standard (non-lifecycle) rclpy.Node for simplicity. If the host
stack needs lifecycle management it can wrap this in a lifecycle node; the core
logic is independent of node type.

Port attribution:
- SafetyEnvelope: HELIX helix_recovery/recovery_node.py:36 (cooldown, allowlist,
  RESUME exempt). See phm_recovery._core for full port commentary.
- cmd_vel hold timer: HELIX helix_recovery/recovery_node.py:142 (_on_publish_tick).
- Actuation pluggability: the host stack calls node.rewind_hook.register(cb) to
  override the rewind callback.
"""

from __future__ import annotations

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from phm_recovery._core import (
    ACTION_HOLD,
    ACTION_LOG_ONLY,
    ACTION_NONE,
    ACTION_REWIND,
    ACTION_STOP_AND_HOLD,
    STATE_OK,
    HealthToActionMapper,
    RewindHook,
    SafetyEnvelope,
)

# QoS for /phm/health subscription: reliable, depth 10.
# Health status is a command-class topic; reliability is required so no status
# message is silently dropped under load.
_HEALTH_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    depth=10,
)

# QoS for /phm/cmd_vel: reliable, depth 10.
# Zero-velocity commands must not be dropped.
_CMD_VEL_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    depth=10,
)


class RecoveryNode(Node):
    """PHM Recovery Node: health status -> safe actuation.

    Parameters (loaded from config/recovery.yaml):
        recovery.enabled (bool, default False): master enable. False -> no
            actuation, envelope suppresses all actions.
        recovery.cooldown_seconds (float, default 5.0): minimum time between
            repeated HOLD/STOP_AND_HOLD actuations for the same fault_key.
        recovery.publish_hz (float, default 20.0): rate at which zero-velocity
            Twist is re-published while a hold is active.
    """

    def __init__(self) -> None:
        super().__init__("phm_recovery_node")

        # Parameters.
        self.declare_parameter(
            "recovery.enabled",
            False,
            descriptor=_make_descriptor(
                "Master enable for actuation. False -> no commands published."
            ),
        )
        self.declare_parameter(
            "recovery.cooldown_seconds",
            5.0,
            descriptor=_make_descriptor(
                "Minimum seconds between repeated HOLD/STOP_AND_HOLD for the same fault_key."
            ),
        )
        self.declare_parameter(
            "recovery.publish_hz",
            20.0,
            descriptor=_make_descriptor(
                "Rate (Hz) at which zero-velocity Twist is re-published while hold is active."
            ),
        )

        enabled: bool = self.get_parameter("recovery.enabled").value
        cooldown: float = self.get_parameter("recovery.cooldown_seconds").value
        publish_hz: float = self.get_parameter("recovery.publish_hz").value

        # Pure-logic objects (no ROS dep).
        self._envelope = SafetyEnvelope(enabled=enabled, cooldown_seconds=cooldown)
        self._mapper = HealthToActionMapper()
        self.rewind_hook = RewindHook()  # public: host stack registers callback here

        # ROS interface.
        try:
            from phm_msgs.msg import PolicyHealthStatus  # type: ignore[import]
            self._sub_health = self.create_subscription(
                PolicyHealthStatus,
                "/phm/health",
                self._on_health,
                _HEALTH_QOS,
            )
        except ImportError:
            self.get_logger().warning(
                "phm_msgs not built yet; /phm/health subscription skipped. "
                "Build phm_msgs with colcon before running this node."
            )
            self._sub_health = None

        self._pub_cmd_vel = self.create_publisher(Twist, "/phm/cmd_vel", _CMD_VEL_QOS)

        # Zero-velocity publish timer. Publishes only when hold is active.
        # Ported from HELIX recovery_node.py:142 (_on_publish_tick).
        period = 1.0 / max(publish_hz, 0.1)  # guard against zero/negative hz
        self._publish_timer = self.create_timer(period, self._on_publish_tick)

        self.get_logger().info(
            f"PHM RecoveryNode started: enabled={enabled} "
            f"cooldown={cooldown}s publish_hz={publish_hz}Hz"
        )

    # ------------------------------------------------------------------
    # Subscription callback
    # ------------------------------------------------------------------

    def _on_health(self, msg) -> None:  # msg: PolicyHealthStatus
        """Process one health status message.

        Delegates to HealthToActionMapper, then gates through SafetyEnvelope.
        """
        now = self._now()
        fault_key = msg.source or "unknown"

        # 1. Map state + suggested_action to an actuation decision.
        decision = self._mapper.map(
            state=int(msg.state),
            suggested_action=int(msg.suggested_action),
            source=msg.source,
            reason=msg.reason,
        )

        # 2. If the state is back to OK/DEGRADED and hold was cleared by the
        #    mapper, signal a RESUME through the envelope so the cooldown is
        #    not applied to the resume itself.
        if not decision.hold_active and int(msg.state) in (STATE_OK, 1):  # 1 = DEGRADED
            result = self._envelope.evaluate_resume(fault_key, now)
            self.get_logger().info(
                f"RESUME: {result.reason}",
                throttle_duration_sec=1.0,
            )
            return

        if decision.action == ACTION_NONE:
            return

        # 3. Gate the decided action through the safety envelope.
        result = self._envelope.evaluate(decision.action, fault_key, now)

        self.get_logger().info(
            f"envelope: action={decision.action} status={result.status} "
            f"reason={result.reason}",
            throttle_duration_sec=1.0,
        )

        if not result.publish:
            return

        # 4. Dispatch actuating actions.
        if decision.action in (ACTION_HOLD, ACTION_STOP_AND_HOLD):
            # Zero-velocity publishing is handled by _on_publish_tick while
            # _mapper.hold_active is True. Nothing else needed here.
            self.get_logger().warning(
                f"HOLD active: {decision.reason}",
                throttle_duration_sec=1.0,
            )
        elif decision.action == ACTION_REWIND:
            self.get_logger().warning(
                f"REWIND triggered: {decision.reason}",
                throttle_duration_sec=1.0,
            )
            self.rewind_hook.trigger()
        elif decision.action == ACTION_LOG_ONLY:
            self.get_logger().info(f"LOG_ONLY: {decision.reason}", throttle_duration_sec=1.0)

    # ------------------------------------------------------------------
    # Timer callback: zero-velocity re-publisher
    # Ported from HELIX recovery_node.py:141-143 (_on_publish_tick).
    # ------------------------------------------------------------------

    def _on_publish_tick(self) -> None:
        if self._mapper.hold_active:
            self._pub_cmd_vel.publish(Twist())  # zero velocity

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9


def _make_descriptor(description: str):
    from rcl_interfaces.msg import ParameterDescriptor  # type: ignore[import]
    d = ParameterDescriptor()
    d.description = description
    d.read_only = False
    return d


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RecoveryNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
