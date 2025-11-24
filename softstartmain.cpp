#include "main.h"
#include "rclcpp/rclcpp.hpp"

/**
 * main - entrance of controller.
 * @param argc
 * @param argv
 * @return 0
 */


int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);     
      
  std::shared_ptr<can_driver> Can_driver_  = std::make_shared<can_driver>();
  std::shared_ptr<lowerproxy> Lower_proxy_ = std::make_shared<lowerproxy>();
  std::shared_ptr<upperproxy> Upper_proxy_ = std::make_shared<upperproxy>();

  ControllerMonitor & monitor = ControllerMonitor::GetStateMonitor();
  TrajectoriesParser & traj_parser = TrajectoriesParser::getTrajParser();

  monitor.Init();
  traj_parser.init();
  Can_driver_->get_motor_status(traveler_leg_);

  // Target joint positions
  const float target_pos_axis0 = M_PI / 4.0f;
  const float target_pos_axis1 = 1.75f * M_PI;

  // Read initial positions
  float start_pos_axis0 = traveler_leg_.traveler_chassis.Leg_lf.axis0.position;
  float start_pos_axis1 = traveler_leg_.traveler_chassis.Leg_lf.axis1.position;

  // Compare and decide if soft start is needed
  const float TOLERANCE = 0.01f;
  float delta_axis0 = std::abs(target_pos_axis0 - start_pos_axis0);
  float delta_axis1 = std::abs(target_pos_axis1 - start_pos_axis1);

  bool need_soft_start = (delta_axis0 > TOLERANCE) || (delta_axis1 > TOLERANCE);

  if (need_soft_start) {
      const int steps = 500;
      const float dt = 5.0f / steps;

      for (int i = 0; i <= steps; ++i) {
          float alpha = static_cast<float>(i) / steps;

          traveler_leg_.traveler_control.Leg_lf.axis0.motor_control_position =
              start_pos_axis0 + alpha * (target_pos_axis0 - start_pos_axis0);
          traveler_leg_.traveler_control.Leg_lf.axis1.motor_control_position =
              start_pos_axis1 + alpha * (target_pos_axis1 - start_pos_axis1);

          Can_driver_->setControl(traveler_leg_);
          std::this_thread::sleep_for(std::chrono::duration<float>(dt));
      }
  } else {
      traveler_leg_.traveler_control.Leg_lf.axis0.motor_control_position = target_pos_axis0;
      traveler_leg_.traveler_control.Leg_lf.axis1.motor_control_position = target_pos_axis1;
      Can_driver_->setControl(traveler_leg_);
  }


  rclcpp::Rate loop_rate(1500);       //renew frequence 100HZ

  while (rclcpp::ok()) {
    rclcpp::spin_some(Upper_proxy_);
    // // remove spin low proxy, because it doesnot need to receive message
    // // rclcpp::spin_some(Lower_proxy_);
    // rclcpp::spin_some(Can_driver_);
    Can_driver_->get_motor_status(traveler_leg_);
    Lower_proxy_->UpdateJoystickStatus(traveler_leg_);             //update leg feedback status
    Upper_proxy_->UpdateGuiCommand(traveler_leg_);             //update gui command
    //printf("traveler extrude angle: %f", traveler_leg_.traj_data.extrude_angle);
    traj_parser.generateTempTraj(traveler_leg_);
    //Lower_proxy_->PublishControlCommand(traveler_leg_);
    Lower_proxy_->calculate_position(traveler_leg_);             
    Can_driver_->setControl(traveler_leg_);
    
    // Upper_proxy_->PublishStatusFeedback(traveler_leg_);             //publish current times
    loop_rate.sleep();
  }


  return 0;
}
