#!/usr/bin/env python3
"""Shadow-mode Phoenix policy on a live Unitree GO2, publishing its latent for PHM.

Runs the Phoenix stand-v3 locomotion policy (ONNX, exported with ``--emit-latent``)
on the robot's real ``/lowstate`` at the policy's trained 50 Hz rate and publishes
the 384-D latent as ``phm_msgs/PolicyEmbedding`` on ``/policy/embedding``.

It never commands the robot. The only publisher in this process is the embedding
topic. The policy's action output is used for one thing: the ``last_action``
observation term of the next tick, exactly as the deploy node feeds it. Because
that action is never applied, the policy runs open loop on its own action term;
the joint, IMU and gravity terms are the robot's real state.

Observation assembly goes through Phoenix's own ``assemble_policy_observation``
(``phoenix.sim2real.observation``), the single sensor-to-policy path its parity
gate checks, so this node cannot drift from the deploy node's obs contract.

Fault injection (for an induced-failure run, still with no actuation):
  fault = "none"        default; no fault.
  fault = "freeze_obs"  after ``fault_after_sec``, keep feeding the policy the
                        last observation, as a policy wired to a stale sensor
                        snapshot would. The latent stops varying.
  fault = "stop"        after ``fault_after_sec``, stop publishing, as a crashed
                        policy process would. The embedding topic goes silent.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

# Unitree motor indices 0..11 (go2-phoenix lowstate_bridge_node.MOTOR_NAMES).
MOTOR_NAMES: tuple[str, ...] = (
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
)

# Policy joint order and training default pose
# (go2-phoenix configs/sim2real/deploy_stand_v3_shielded.yaml).
POLICY_JOINT_ORDER: tuple[str, ...] = (
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
)
DEFAULT_JOINT_POS: dict[str, float] = {
    "FL_hip_joint": 0.1, "FR_hip_joint": -0.1, "RL_hip_joint": 0.1, "RR_hip_joint": -0.1,
    "FL_thigh_joint": 0.8, "FR_thigh_joint": 0.8, "RL_thigh_joint": 1.0, "RR_thigh_joint": 1.0,
    "FL_calf_joint": -1.5, "FR_calf_joint": -1.5, "RL_calf_joint": -1.5, "RR_calf_joint": -1.5,
}

FAULTS = ("none", "freeze_obs", "stop")


def motor_to_policy_index() -> np.ndarray:
    """Index array that reorders Unitree motor-order vectors into policy order."""
    lookup = {name: i for i, name in enumerate(MOTOR_NAMES)}
    return np.asarray([lookup[n] for n in POLICY_JOINT_ORDER], dtype=np.int64)


def unitree_quat_to_xyzw(q_wxyz) -> tuple[float, float, float, float]:
    """Unitree IMUState.quaternion is [w, x, y, z]; Phoenix takes (x, y, z, w)."""
    return float(q_wxyz[1]), float(q_wxyz[2]), float(q_wxyz[3]), float(q_wxyz[0])


class PhoenixShadowEmbedder(Node):
    def __init__(self) -> None:
        super().__init__("phoenix_shadow_embedder")
        self.declare_parameter("onnx_path", "")
        self.declare_parameter("phoenix_src", "")
        self.declare_parameter("rate_hz", 50.0)
        self.declare_parameter("embedding_topic", "/policy/embedding")
        self.declare_parameter("policy_id", "phoenix-stand-v3")
        self.declare_parameter("fault", "none")
        self.declare_parameter("fault_after_sec", -1.0)
        self.declare_parameter("stats_every_sec", 10.0)

        onnx_path = self.get_parameter("onnx_path").value
        phoenix_src = self.get_parameter("phoenix_src").value
        self._fault = self.get_parameter("fault").value
        self._fault_after = float(self.get_parameter("fault_after_sec").value)
        self._policy_id = self.get_parameter("policy_id").value
        if self._fault not in FAULTS:
            raise ValueError(f"fault must be one of {FAULTS}, got {self._fault!r}")
        if not onnx_path or not os.path.isfile(onnx_path):
            raise FileNotFoundError(f"onnx_path {onnx_path!r} does not exist")
        if phoenix_src:
            sys.path.insert(0, phoenix_src)

        import onnxruntime as ort  # noqa: PLC0415
        from phoenix.sim2real.observation import (  # noqa: PLC0415
            JointOrder,
            ObservationBuilder,
            assemble_policy_observation,
        )
        from unitree_go.msg import LowState  # noqa: PLC0415

        from phm_msgs.msg import PolicyEmbedding  # noqa: PLC0415

        self._PolicyEmbedding = PolicyEmbedding
        self._assemble = assemble_policy_observation
        self._builder = ObservationBuilder(JointOrder(POLICY_JOINT_ORDER), DEFAULT_JOINT_POS)

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        self._session = ort.InferenceSession(
            onnx_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        out_names = [o.name for o in self._session.get_outputs()]
        if "latent" not in out_names or "action" not in out_names:
            raise RuntimeError(
                f"{onnx_path} outputs {out_names}; need 'action' and 'latent' "
                "(export with phoenix.sim2real.export --emit-latent)"
            )
        self._latent_dim = int(self._session.get_outputs()[out_names.index("latent")].shape[-1])

        self._idx = motor_to_policy_index()
        self._last_action = np.zeros(12, dtype=np.float32)
        self._zeros3 = np.zeros(3, dtype=np.float32)
        self._latest = None  # (joint_pos, joint_vel, quat_xyzw, gyro), policy order
        self._latest_rx = 0.0
        self._frozen_obs: np.ndarray | None = None
        self._fault_logged = False
        self._t0 = time.monotonic()
        self._infer_ms: list[float] = []
        self._tick_ms: list[float] = []
        self._n_pub = 0
        self._stats_t = self._t0

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._sub = self.create_subscription(LowState, "/lowstate", self._on_lowstate, sensor_qos)
        self._pub = self.create_publisher(
            PolicyEmbedding, self.get_parameter("embedding_topic").value, sensor_qos
        )
        period = 1.0 / float(self.get_parameter("rate_hz").value)
        self._timer = self.create_timer(period, self._tick)
        self.get_logger().info(
            f"shadow policy {self._policy_id} ({onnx_path}) latent_dim={self._latent_dim} "
            f"at {1.0 / period:.0f} Hz; fault={self._fault} "
            f"after {self._fault_after:.1f}s. NO actuation: only publisher is the embedding."
        )

    def _on_lowstate(self, msg) -> None:
        q = np.fromiter((m.q for m in msg.motor_state[:12]), dtype=np.float32, count=12)
        dq = np.fromiter((m.dq for m in msg.motor_state[:12]), dtype=np.float32, count=12)
        self._latest = (
            q[self._idx],
            dq[self._idx],
            unitree_quat_to_xyzw(msg.imu_state.quaternion),
            np.asarray(msg.imu_state.gyroscope, dtype=np.float32),
        )
        self._latest_rx = time.monotonic()

    def _fault_active(self, now: float) -> bool:
        if self._fault == "none" or self._fault_after < 0:
            return False
        return now - self._t0 >= self._fault_after

    def _tick(self) -> None:
        t_start = time.perf_counter()
        now = time.monotonic()
        if self._latest is None:
            return
        fault_on = self._fault_active(now)
        if fault_on and self._fault == "stop":
            if not self._fault_logged:
                self.get_logger().warn("FAULT INJECTED: stop (policy stops publishing)")
                self._fault_logged = True
            return

        if fault_on and self._fault == "freeze_obs" and self._frozen_obs is not None:
            obs = self._frozen_obs
        else:
            joint_pos, joint_vel, quat_xyzw, gyro = self._latest
            obs = self._assemble(
                self._builder,
                base_lin_vel=self._zeros3,  # stand-v3 calibrated distribution (zeros source)
                quat_xyzw=quat_xyzw,
                base_ang_vel=gyro,
                velocity_command=self._zeros3,
                joint_pos=joint_pos,
                joint_vel=joint_vel,
                last_action=self._last_action,
            ).reshape(1, -1)
            if fault_on and self._fault == "freeze_obs":
                self._frozen_obs = obs
                self.get_logger().warn("FAULT INJECTED: freeze_obs (policy sees a stale snapshot)")

        t_inf = time.perf_counter()
        action, latent = self._session.run(["action", "latent"], {"obs": obs})
        self._infer_ms.append((time.perf_counter() - t_inf) * 1e3)
        self._last_action = action[0].astype(np.float32, copy=False)

        msg = self._PolicyEmbedding()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.embedding = latent[0].astype(np.float32).tolist()
        msg.dim = self._latent_dim
        msg.policy_id = self._policy_id
        self._pub.publish(msg)
        self._n_pub += 1
        self._tick_ms.append((time.perf_counter() - t_start) * 1e3)

        if now - self._stats_t >= float(self.get_parameter("stats_every_sec").value):
            inf = np.asarray(self._infer_ms)
            tick = np.asarray(self._tick_ms)
            self.get_logger().info(
                f"published {self._n_pub}; onnx p50 {np.percentile(inf, 50):.3f} ms "
                f"p99 {np.percentile(inf, 99):.3f} ms; tick p50 {np.percentile(tick, 50):.3f} ms "
                f"p99 {np.percentile(tick, 99):.3f} ms; lowstate age "
                f"{(now - self._latest_rx) * 1e3:.1f} ms"
            )
            self._infer_ms.clear()
            self._tick_ms.clear()
            self._stats_t = now


def main() -> int:
    rclpy.init()
    node = PhoenixShadowEmbedder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
