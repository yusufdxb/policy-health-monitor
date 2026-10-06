// Copyright 2026 Yusuf Guenena. MIT License.
// ros2 run phm_detectors phm_detectors_node
#include <memory>

#include "phm_detectors/detectors_node.hpp"
#include "rclcpp/rclcpp.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<phm_detectors::DetectorsNode>());
  rclcpp::shutdown();
  return 0;
}
