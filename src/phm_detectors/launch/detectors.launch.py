"""Launch file for the phm_detectors node.

Loads parameters from config/detectors.yaml and starts the
phm_detectors_node which publishes DetectorVerdict to /phm/verdicts.
"""

from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node

from launch import LaunchDescription


def generate_launch_description() -> LaunchDescription:
    config = os.path.join(
        get_package_share_directory("phm_detectors"),
        "config",
        "detectors.yaml",
    )

    detectors_node = Node(
        package="phm_detectors",
        executable="phm_detectors_node",
        name="phm_detectors",
        output="screen",
        parameters=[config],
    )

    return LaunchDescription([detectors_node])
