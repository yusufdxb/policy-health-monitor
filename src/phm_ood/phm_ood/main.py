"""Entry point for the phm_ood lifecycle node."""

from __future__ import annotations


def main(args=None):  # type: ignore[no-untyped-def]
    """Spin the OodNode as a standalone process."""
    import rclpy
    from rclpy.executors import SingleThreadedExecutor

    from phm_ood.node import OodNode

    rclpy.init(args=args)
    node = OodNode()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
