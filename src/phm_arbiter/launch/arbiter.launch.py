"""Launch the phm_arbiter node with the default parameter YAML.

Usage:
    ros2 launch phm_arbiter arbiter.launch.py
    ros2 launch phm_arbiter arbiter.launch.py staleness_sec:=2.0
"""

from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from launch import LaunchDescription


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory("phm_arbiter")
    default_config = os.path.join(pkg_share, "config", "phm_arbiter.yaml")

    staleness_arg = DeclareLaunchArgument(
        "staleness_sec",
        default_value="1.0",
        description="Verdict staleness threshold in seconds.",
    )
    timer_arg = DeclareLaunchArgument(
        "timer_period",
        default_value="0.05",
        description="Arbitration timer period in seconds (20 Hz default).",
    )

    arbiter_node = Node(
        package="phm_arbiter",
        executable="phm_arbiter",
        name="phm_arbiter",
        output="screen",
        parameters=[
            default_config,
            {
                "staleness_sec": LaunchConfiguration("staleness_sec"),
                "timer_period": LaunchConfiguration("timer_period"),
            },
        ],
    )

    return LaunchDescription([staleness_arg, timer_arg, arbiter_node])
