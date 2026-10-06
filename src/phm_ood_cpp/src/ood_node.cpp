// Copyright 2026 Yusuf Guenena. MIT License.
// ros2 run phm_ood_cpp ood_node: the C++ rolling-spread OOD lifecycle node
// (node "phm_ood_cpp", verdict source "phm_ood_cpp"). It starts unconfigured;
// configure and activate it through the lifecycle services.
#include <memory>

#include "phm_ood_cpp/ood_lifecycle_node.hpp"
#include "rclcpp/rclcpp.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::executors::SingleThreadedExecutor executor;
  auto node = std::make_shared<phm_ood_cpp::OodLifecycleNode>(
    phm_ood_cpp::OodNodeProfile::phm_ood_cpp());
  executor.add_node(node->get_node_base_interface());
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
