// Copyright 2026 Yusuf Guenena. MIT License.
// ros2 run phm_ood phm_ood_node: the rolling-spread OOD lifecycle node with the
// phm_ood interface (node "phm_ood", verdict source "phm_ood", parameters from
// config/phm_ood.yaml). It runs the same implementation as phm_ood_cpp's
// ood_node; see phm_ood_cpp/ood_lifecycle_node.hpp for the profile
// differences. It starts unconfigured; configure and activate it through the
// lifecycle services.
#include <memory>

#include "phm_ood_cpp/ood_lifecycle_node.hpp"
#include "rclcpp/rclcpp.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::executors::SingleThreadedExecutor executor;
  auto node = std::make_shared<phm_ood_cpp::OodLifecycleNode>(
    phm_ood_cpp::OodNodeProfile::phm_ood());
  executor.add_node(node->get_node_base_interface());
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
