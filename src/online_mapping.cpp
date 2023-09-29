#include <ros/ros.h>
#include "Mapping.hpp"
bool is_exit = false;

void SigHandle(int sig)
{
    ROS_WARN("catch sig %d", sig);
    is_exit = true;
}
int main(int argc, char **argv)
{
    ros::init(argc, argv, "laserMapping");
    ros::NodeHandle nh;

    auto laser_mapping = std::make_shared<LaserMapping>();
    laser_mapping->initOnline(nh);

    signal(SIGINT, SigHandle);
    ros::Rate rate(5000);

    // online, almost same with offline, just receive the messages from ros
    while (ros::ok())
    {
        if (is_exit)
        {
            break;
        }
        ros::spinOnce();
        laser_mapping->RunOnce();
        rate.sleep();
    }

    ROS_INFO("laserMapping online is exiting");
    return 0;
}