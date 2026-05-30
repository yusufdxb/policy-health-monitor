"""Launch file for the phm_ood OOD detector node.

Starts a single OodNode lifecycle node, loading parameters from
config/phm_ood.yaml. The node starts in the unconfigured state; send a
lifecycle Configure then Activate transition to begin detection.

Usage:
    ros2 launch phm_ood phm_ood.launch.py
    ros2 launch phm_ood phm_ood.launch.py threshold:=0.019 window:=30

Override any parameter at the command line; the YAML file provides defaults.
"""

from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode

from launch import LaunchDescription


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory("phm_ood")
    default_params = os.path.join(pkg_share, "config", "phm_ood.yaml")

    # Expose key parameters as launch arguments.
    args = [
        DeclareLaunchArgument(
            "embedding_topic",
            default_value="/policy/embedding",
            description="Topic name for PolicyEmbedding input.",
        ),
        DeclareLaunchArgument(
            "window",
            default_value="30",
            description="Rolling-spread window length (frames, int).",
        ),
        DeclareLaunchArgument(
            "threshold",
            default_value="0.0",
            description="Pre-calibrated spread threshold (float).",
        ),
        DeclareLaunchArgument(
            "calibration_file",
            default_value="",
            description="Path to .npz calibration file (overrides threshold param).",
        ),
        DeclareLaunchArgument(
            "hysteresis_count",
            default_value="3",
            description="Consecutive violating frames before verdict flips.",
        ),
        DeclareLaunchArgument(
            "compute_every",
            default_value="1",
            description="Compute spread every N frames (frequency gate).",
        ),
        DeclareLaunchArgument(
            "embed_dim",
            default_value="0",
            description="Expected embedding dim for validation. 0=skip.",
        ),
    ]

    node = LifecycleNode(
        package="phm_ood",
        executable="phm_ood_node",
        name="phm_ood",
        namespace="",
        output="screen",
        parameters=[
            default_params,
            {
                "embedding_topic": LaunchConfiguration("embedding_topic"),
                "window": LaunchConfiguration("window"),
                "threshold": LaunchConfiguration("threshold"),
                "calibration_file": LaunchConfiguration("calibration_file"),
                "hysteresis_count": LaunchConfiguration("hysteresis_count"),
                "compute_every": LaunchConfiguration("compute_every"),
                "embed_dim": LaunchConfiguration("embed_dim"),
            },
        ],
    )

    return LaunchDescription(args + [node])
