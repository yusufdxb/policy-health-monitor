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
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)

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

# QoS for /phm/health subscription (review decision 7): all four policies
# declared explicitly. durability MUST be TRANSIENT_LOCAL to match the arbiter
# publisher (arbiter_node.py health_qos); a VOLATILE subscriber against a
# TRANSIENT_LOCAL publisher is QoS-incompatible and silently drops every health
# message. Health status is a command-class topic, so reliability is RELIABLE.
_HEALTH_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)

# QoS for /phm/cmd_vel (review decision 7): all four policies explicit.
# Zero-velocity commands must not be dropped (RELIABLE); cmd_vel is a live
# stream so durability is VOLATILE (a late joiner does not need a stale stop).
_CMD_VEL_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
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

        # Hold actuation flag. UNLIKE the mapper's own hold_active, this is the
        # ENVELOPE-coupled actuation state: it is set True only when the envelope
        # returns publish=True for a hold, and cleared only on a real RESUME
        # (LOCKED decision 5). The zero-velocity timer reads THIS flag, not the
        # mapper's, so cooldown gating actually couples to what gets published.
        self._hold_actuating: bool = False

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

        Delegates to HealthToActionMapper, then couples hold actuation to the
        SafetyEnvelope result (LOCKED decision 5, mirroring HELIX
        recovery_node.py:133-138):

        - The mapper proposes an action and whether a hold should be active.
        - A hold actuates (``self._hold_actuating`` -> True, driving the
          zero-velocity timer) only when the envelope returns publish=True.
        - Cooldown damps NEW holds only. A continued hold is cooldown-exempt
          (passed via hold_already_active), so SUPPRESSED_COOLDOWN can never
          release an ongoing hold. A genuine recovery (OK/DEGRADED) clears it.
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

        # 2. Recovery: the state returned to OK/DEGRADED and the mapper cleared
        #    the hold. Release the actuation via a cooldown-exempt RESUME.
        if not decision.hold_active and int(msg.state) in (STATE_OK, 1):  # 1 = DEGRADED
            result = self._envelope.evaluate_resume(fault_key, now)
            if result.publish:
                self._hold_actuating = False
            self.get_logger().info(
                f"RESUME: {result.reason}",
                throttle_duration_sec=1.0,
            )
            return

        if decision.action == ACTION_NONE:
            return

        # 3. Gate the decided action through the safety envelope. A hold that is
        #    already actuating is a cooldown-exempt continuation.
        result = self._envelope.evaluate(
            decision.action,
            fault_key,
            now,
            hold_already_active=self._hold_actuating
            and decision.action in (ACTION_HOLD, ACTION_STOP_AND_HOLD, ACTION_REWIND),
        )

        self.get_logger().info(
            f"envelope: action={decision.action} status={result.status} "
            f"reason={result.reason}",
            throttle_duration_sec=1.0,
        )

        # 4. Couple hold actuation to the envelope (HELIX recovery_node.py:133-138).
        #    A NEW hold actuates only when the envelope says publish. A
        #    SUPPRESSED_COOLDOWN on a re-assert does NOT clear an ongoing hold:
        #    the hold keeps publishing because _hold_actuating stays True.
        if not result.publish:
            return

        if decision.action in (ACTION_HOLD, ACTION_STOP_AND_HOLD):
            self._hold_actuating = True
            self.get_logger().warning(
                f"HOLD active: {decision.reason}",
                throttle_duration_sec=1.0,
            )
        elif decision.action == ACTION_REWIND:
            # REWIND seam (LOCKED decision 4): invoke the rewind hook AND hold.
            self._hold_actuating = True
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
        # Drive zero-velocity off the ENVELOPE-coupled actuation flag, not the
        # mapper's independent hold flag (LOCKED decision 5). This is what makes
        # the cooldown gate meaningful: the hold publishes iff the envelope
        # actuated it and it has not been released by a RESUME.
        if self._hold_actuating:
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
