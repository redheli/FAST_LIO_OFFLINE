
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <unistd.h>
#include <csignal>
#include <signal.h>
#include <boost/stacktrace.hpp>
#include <future>

#include "Mapping.hpp"
#include <livox_ros_driver/CustomMsg.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Imu.h>


std::atomic<bool> is_exit(false);

void print_stacktrace(int signum)
{
    ::signal(signum, SIG_DFL);
    std::cerr << "Stack trace:\n"
              << boost::stacktrace::stacktrace() << '\n';
    ::raise(SIGABRT);
}

void SigHandle(int sig)
{
    ROS_WARN("catch sig %d", sig);
    is_exit = true;
}

struct MessageHolder
{
    MessageHolder()
        : imu_msg(nullptr),
          livox_msg(nullptr),
          point_cloud_msg(nullptr) {}
    MessageHolder(const MessageHolder &other)
        : imu_msg(other.imu_msg),
          livox_msg(other.livox_msg),
          point_cloud_msg(other.point_cloud_msg) {}
    sensor_msgs::Imu::Ptr imu_msg;
    livox_ros_driver::CustomMsg::ConstPtr livox_msg;
    sensor_msgs::PointCloud2::ConstPtr point_cloud_msg;
    double getTime() const
    {
        if (imu_msg)
        {
            return imu_msg->header.stamp.toSec();
        }
        if (livox_msg)
        {
            return livox_msg->header.stamp.toSec();
        }
        if (point_cloud_msg)
        {
            return point_cloud_msg->header.stamp.toSec();
        }
        throw std::runtime_error("MessageHolder: no message");
        return 0;
    }
};

// Comparator for sorting messages by timestamp
struct CompareTimestamp
{
    bool operator()(MessageHolder const &m1, MessageHolder const &m2)
    {
        // Reverse the order for oldest message first
        return m1.getTime() > m2.getTime();
    }
};

int main(int argc, char **argv)
{
    ::signal(SIGSEGV, &print_stacktrace);
    ::signal(SIGABRT, &print_stacktrace);

    if (argc < 3)
    {
        ROS_ERROR_STREAM("Usage: rosrun offline_mapping <bag_file> <config_file>");
        return -1;
    }

    const std::string bag_file = argv[1];
    const std::string config_file = argv[2];


    ROS_INFO_STREAM("bag_file: " << bag_file);
    ROS_INFO_STREAM("config_file: " << config_file);

    // get rosbag name and assign to laser_mapping prefix
    std::string bag_name = bag_file.substr(bag_file.find_last_of("/") + 1);
    bag_name = bag_name.substr(0, bag_name.find_last_of("."));
    std::cout << "bag_name: " << bag_name << std::endl;
    ROS_INFO_STREAM("bag_file: " << bag_file);

    auto laser_mapping = std::make_shared<LaserMapping>(bag_name);
    laser_mapping->initWithoutROS(config_file);
    
    ROS_INFO_STREAM("LaserMapping init OK");

    std::promise<void> exitSignal;
    std::future<void> futureObj = exitSignal.get_future();


    std::thread runOnceThread([&]()
                              {
    while (!is_exit) {
        laser_mapping->RunOnce();
        std::this_thread::sleep_for(std::chrono::milliseconds(10)); // equivalent to usleep(1000);
    }
        exitSignal.set_value(); // Signal that the thread has finished
    });

    runOnceThread.detach();

    /// handle ctrl-c
    signal(SIGINT, SigHandle);

    // just read the bag and send the data
    ROS_INFO_STREAM("Opening rosbag ...");
    rosbag::Bag bag(bag_file, rosbag::bagmode::Read);
    ROS_INFO_STREAM("Open rosbag OK");

    ROS_INFO_STREAM("Processing bag file ..."<<bag_file);
    std::priority_queue<MessageHolder, std::vector<MessageHolder>, CompareTimestamp> messageQueue;

    int pc_count = 0;
    double last_lidar_time = 0;
    int imu_count = 0;
    double first_message_ts = 0;

    rosbag::View view(bag);
    ros::Time bag_begin_time = view.getBeginTime();
    ros::Time bag_end_time = view.getEndTime();
    double total_time = bag_end_time.toSec() - bag_begin_time.toSec();
    for (const rosbag::MessageInstance &msg : view)
    {
        if (is_exit)
        {
            ROS_INFO_STREAM("exit, queue size " << messageQueue.size());
            break;
        }
       
        {
            // add to queue
            MessageHolder holder;
            auto livox_msg = msg.instantiate<livox_ros_driver::CustomMsg>();
            if (livox_msg && msg.getTopic() == laser_mapping->lid_topic)
            {
                livox_ros_driver::CustomMsg::Ptr new_msg(new livox_ros_driver::CustomMsg(*livox_msg));
                holder.livox_msg = new_msg;
                messageQueue.push(holder);
            }
            auto point_cloud_msg = msg.instantiate<sensor_msgs::PointCloud2>();
            if (point_cloud_msg && msg.getTopic() == laser_mapping->lid_topic)
            {
                sensor_msgs::PointCloud2::Ptr new_msg(new sensor_msgs::PointCloud2(*point_cloud_msg));
                holder.point_cloud_msg = new_msg;
                messageQueue.push(holder);
            }
            auto imu_msg = msg.instantiate<sensor_msgs::Imu>();
            if (imu_msg && msg.getTopic() == laser_mapping->imu_topic)
            {
                imu_count++;
                sensor_msgs::Imu::Ptr new_msg(new sensor_msgs::Imu(*imu_msg));
                holder.imu_msg = new_msg;
                messageQueue.push(holder);
            }
        }
        if (messageQueue.size() < 100)
        {
            continue;
        }

        const MessageHolder msg_holder = messageQueue.top();
        messageQueue.pop();

        if (msg_holder.livox_msg)
        {
            ROS_INFO_STREAM("process livox_msg " <<std::fixed << std::setprecision(2) << msg_holder.livox_msg->header.stamp.toSec() - first_message_ts<<"/"<<total_time<<"s\n");
            double diff = msg_holder.livox_msg->header.stamp.toSec() - last_lidar_time;
            last_lidar_time = msg_holder.livox_msg->header.stamp.toSec();
            laser_mapping->livox_pcl_cbk(msg_holder.livox_msg);
            // sleep 100ms
            usleep(100000);
            pc_count++;
            continue;
        }

        if (msg_holder.point_cloud_msg)
        {
            ROS_INFO_STREAM("process point_cloud_msg " << std::fixed << std::setprecision(2) << msg_holder.point_cloud_msg->header.stamp.toSec() - first_message_ts<<"/"<<total_time<<"s\n");
            laser_mapping->standard_pcl_cbk(msg_holder.point_cloud_msg);
            usleep(100000);
            pc_count++;
            continue;
        }

        if (msg_holder.imu_msg)
        {
            laser_mapping->imu_cbk(msg_holder.imu_msg);
            if (first_message_ts == 0)
            {
                first_message_ts = msg_holder.imu_msg->header.stamp.toSec();
            }
            continue;
        }
    }

    ROS_INFO_STREAM("Finished processing bag file.");
    is_exit = true;
    futureObj.wait(); // Wait for the signal from the detached thread
    if (pc_count == 0)
    {
        ROS_ERROR_STREAM("No point cloud messages processed. Exiting.");
        return -1;
    }
    // save map to pcd
    laser_mapping->savePCD();

    return 0;
}
