"""Launch file for phm_recovery_node.

Loads config/recovery.yaml so all parameters are declared at launch time.
Set recovery.enabled:=true on the command line to activate actuation:

  ros2 launch phm_recovery recovery.launch.py recovery.enabled:=true
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from launch import LaunchDescription


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory("phm_recovery")
    default_config = os.path.join(pkg_share, "config", "recovery.yaml")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_file",
                default_value=default_config,
                description="Path to the recovery node YAML config.",
            ),
            Node(
                package="phm_recovery",
                executable="recovery_node",
                name="phm_recovery_node",
                parameters=[LaunchConfiguration("config_file")],
                output="screen",
            ),
        ]
    )
