"""Launch file for phm_sim: the embedding replay/perturbation publisher.

Brings up the EmbeddingPublisherNode with parameters loaded from
config/phm_sim.yaml. Intended for local smoke tests and integration runs
that need a PolicyEmbedding source without a real policy process.

Usage:
    ros2 launch phm_sim sim.launch.py

Optional argument overrides (via ros2 launch ... key:=value):
    dim            (int)   override embedding dimensionality
    n_in_dist      (int)   override in-distribution frame count
    in_dist_scale  (float) override healthy phase std-dev
    ood_scale      (float) override OOD phase std-dev
    publish_rate_hz (float) override publish rate
    policy_id      (str)   override policy label
    seed           (int)   override RNG seed
"""

from __future__ import annotations

from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node

from launch import LaunchDescription


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory("phm_sim")
    params_file = f"{pkg_share}/config/phm_sim.yaml"

    embedding_publisher = Node(
        package="phm_sim",
        executable="embedding_publisher",
        name="phm_sim",
        output="screen",
        parameters=[params_file],
    )

    return LaunchDescription([embedding_publisher])
