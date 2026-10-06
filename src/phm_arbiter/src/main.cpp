// Copyright 2026 Yusuf Guenena. MIT License.
// ros2 run phm_arbiter phm_arbiter
#include <memory>

#include "phm_arbiter/arbiter_node.hpp"
#include "rclcpp/rclcpp.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<phm_arbiter::ArbiterNode>());
  rclcpp::shutdown();
  return 0;
}
