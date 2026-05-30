"""phm_sim: replay/perturbation publisher for the Policy Health Monitor.

Publishes ``phm_msgs/PolicyEmbedding`` on ``/policy/embedding``, cycling
through an in-distribution phase followed by a collapse (OOD) phase. Enables
end-to-end smoke tests of the OOD pipeline without a real policy process.

The pure-Python generator is in :mod:`phm_sim._sim_core` (no rclpy). The rclpy
node is in :mod:`phm_sim.embedding_publisher_node`.
"""

from __future__ import annotations

from phm_sim._sim_core import EmbeddingStream, generate_embeddings

__all__ = ["EmbeddingStream", "generate_embeddings"]
