// Copyright 2026 Yusuf Guenena. MIT License.
// ros2 run phm_recovery recovery_node
#include <memory>

#include "phm_recovery/recovery_node.hpp"
#include "rclcpp/rclcpp.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<phm_recovery::RecoveryNode>());
  rclcpp::shutdown();
  return 0;
}
