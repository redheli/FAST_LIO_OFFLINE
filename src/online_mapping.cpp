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
    std::cout << "debug 000000" << std::endl;
    ros::init(argc, argv, "laserMapping");
    std::cout << "debug -11" << std::endl;
    ros::NodeHandle nh;
    std::cout << "debug -12" << std::endl;

    auto laser_mapping = std::make_shared<LaserMapping>();
    std::cout << "debug 0" << std::endl;
    laser_mapping->initOnline(nh);

    signal(SIGINT, SigHandle);
    ros::Rate rate(5000);
    std::cout << "debug 1" << std::endl;

    // online, almost same with offline, just receive the messages from ros
    while (ros::ok())
    {
        if (is_exit)
        {
            break;
        }
        ros::spinOnce();
        laser_mapping->RunOnce();
        // std::cout << "debug 2" << std::endl;
        rate.sleep();
    }

    ROS_INFO("laserMapping online is exiting");
    return 0;
}