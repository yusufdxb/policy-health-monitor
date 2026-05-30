"""rclpy publisher node: streams PolicyEmbedding on /policy/embedding.

This node is a thin adapter over :mod:`phm_sim._sim_core`. All computation
lives in the pure-Python :class:`phm_sim.EmbeddingStream`; this module only
wires the ROS 2 interface (parameters, timer, publisher, trigger service).

Node name: ``phm_sim``
Executable: ``embedding_publisher`` (registered in setup.py)

Topics
------
/policy/embedding (phm_msgs/PolicyEmbedding)
    QoS: reliable, depth=10, keep-last.
    Reason: downstream OOD node needs every frame; best-effort risks dropping
    frames that drive hysteresis counts, causing false misses. Depth 10
    handles any brief subscriber-side back-pressure.

Services
--------
/phm_sim/trigger_ood (std_srvs/Trigger)
    Immediately switches the stream into the OOD phase. Useful for scripted
    smoke tests that do not want to wait for n_in_dist frames.

Parameters (read from config/phm_sim.yaml)
------------------------------------------
dim              int   64     Embedding dimensionality.
n_in_dist        int   100    Frames in the in-distribution phase.
in_dist_scale    float 1.0    Gaussian std-dev for the healthy phase.
ood_scale        float 0.01   Gaussian std-dev for the OOD phase.
publish_rate_hz  float 10.0   Publish frequency in Hz.
policy_id        str   phm_sim  Label forwarded to PolicyEmbedding.policy_id.
seed             int   42     RNG seed for reproducibility.
"""

from __future__ import annotations

import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from std_srvs.srv import Trigger

from phm_sim._sim_core import EmbeddingStream


class EmbeddingPublisherNode(Node):
    """Publishes PolicyEmbedding frames on /policy/embedding.

    Cycles: n_in_dist frames of healthy (high-spread) embeddings, then an
    infinite OOD (collapsed) phase, or switches immediately on a
    /phm_sim/trigger_ood service call.
    """

    def __init__(self) -> None:
        super().__init__("phm_sim")

        # Parameters. All loaded from config/phm_sim.yaml via the launch file.
        self.declare_parameter(
            "dim",
            64,
            ParameterDescriptor(description="Embedding dimensionality.", read_only=True),
        )
        self.declare_parameter(
            "n_in_dist",
            100,
            ParameterDescriptor(
                description="Number of in-distribution frames before switching to OOD phase.",
                read_only=True,
            ),
        )
        self.declare_parameter(
            "in_dist_scale",
            1.0,
            ParameterDescriptor(
                description="Gaussian std-dev for the in-distribution (healthy) phase.",
                read_only=True,
            ),
        )
        self.declare_parameter(
            "ood_scale",
            0.01,
            ParameterDescriptor(
                description="Gaussian std-dev for the OOD (collapse) phase.",
                read_only=True,
            ),
        )
        self.declare_parameter(
            "publish_rate_hz",
            10.0,
            ParameterDescriptor(
                description="Publish frequency in Hz.", read_only=False
            ),
        )
        self.declare_parameter(
            "policy_id",
            "phm_sim",
            ParameterDescriptor(
                description="Label forwarded to PolicyEmbedding.policy_id.",
                read_only=True,
            ),
        )
        self.declare_parameter(
            "seed",
            42,
            ParameterDescriptor(
                description="NumPy RNG seed for reproducibility.", read_only=True
            ),
        )

        dim = int(self.get_parameter("dim").value)
        n_in_dist = int(self.get_parameter("n_in_dist").value)
        in_dist_scale = float(self.get_parameter("in_dist_scale").value)
        ood_scale = float(self.get_parameter("ood_scale").value)
        policy_id = str(self.get_parameter("policy_id").value)
        seed = int(self.get_parameter("seed").value)
        rate_hz = float(self.get_parameter("publish_rate_hz").value)

        self._stream = EmbeddingStream(
            dim=dim,
            n_in_dist=n_in_dist,
            in_dist_scale=in_dist_scale,
            ood_scale=ood_scale,
            policy_id=policy_id,
            seed=seed,
        )

        # Deferred import: phm_msgs may not be built when unit-testing phm_core.
        # The node module is never imported by pure-Python tests (they import
        # phm_sim._sim_core directly), so this import is safe here.
        from phm_msgs.msg import PolicyEmbedding  # noqa: PLC0415

        # QoS: reliable, depth 10. Command-class topic (feeds OOD detector
        # which drives the arbiter); losing frames can cause false misses.
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            depth=10,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self._pub = self.create_publisher(PolicyEmbedding, "/policy/embedding", qos)

        # Service: trigger OOD phase early (for scripted smoke tests).
        self._trigger_srv = self.create_service(
            Trigger,
            "/phm_sim/trigger_ood",
            self._handle_trigger_ood,
        )

        period_s = 1.0 / rate_hz
        self._timer = self.create_timer(period_s, self._timer_cb)

        self.get_logger().info(
            f"phm_sim ready: dim={dim}, n_in_dist={n_in_dist}, "
            f"in_dist_scale={in_dist_scale}, ood_scale={ood_scale}, "
            f"rate={rate_hz} Hz, policy_id={policy_id!r}"
        )

    def _timer_cb(self) -> None:
        """Publish the next embedding frame."""
        from phm_msgs.msg import PolicyEmbedding  # noqa: PLC0415

        vec = self._stream.next_frame()
        msg = PolicyEmbedding()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.embedding = vec.tolist()
        msg.dim = len(vec)
        msg.policy_id = self._stream.policy_id
        self._pub.publish(msg)

        # Throttled log: one line per second maximum to avoid spamming.
        phase = "OOD" if self._stream.is_ood_phase else "in_dist"
        self.get_logger().info(
            f"[phm_sim] frame={self._stream.frame_index} phase={phase}",
            throttle_duration_sec=1.0,
        )

    def _handle_trigger_ood(
        self,
        _request: Trigger.Request,
        response: Trigger.Response,
    ) -> Trigger.Response:
        """Service handler: force-switch to OOD phase."""
        was_already = self._stream.is_ood_phase
        self._stream.trigger_ood()
        msg = (
            "OOD phase was already active."
            if was_already
            else f"Switched to OOD phase at frame {self._stream.frame_index}."
        )
        self.get_logger().info(f"[phm_sim] trigger_ood: {msg}")
        response.success = True
        response.message = msg
        return response


def main(args: list[str] | None = None) -> None:
    """Entry point registered in setup.py."""
    rclpy.init(args=args)
    node = EmbeddingPublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
