// This is an advanced implementation of the algorithm described in the
// following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

// Modifier: Livox               dev@livoxtech.com

// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <fstream>
#include <csignal>
#include <unistd.h>
#include <Python.h>
#include <so3_math.h>
#include <Eigen/Core>

#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/Vector3.h>
// #include <livox_ros_driver/CustomMsg.h>

#include "IMU_Processing.hpp"
#include "preprocess.h"
#include <ikd-Tree/ikd_Tree.h>

#define INIT_TIME (0.1)
#define LASER_POINT_COV (0.001)
#define MAXN (720000)
#define PUBFRAME_PERIOD (20)

class LaserMapping
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    LaserMapping(std::string save_folder_prefix = "");
    ~LaserMapping()
    {
        pose_fs.close();
    }

    /// init without ros
    void initWithoutROS(const std::string &config_yaml);
    void LoadParamsFromYAML(const std::string &yaml);

    void initOnline(ros::NodeHandle &nh);
    void initOthers();

    // void livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg);
    void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg);
    void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in);
    void RunOnce();
    void savePCD();
    bool sync_packages();
    void h_share_model(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data);
    void lasermap_fov_segment();
    void map_incremental();
    void publish_frame_world(const ros::Publisher &pubLaserCloudFull);
    void savePoseAndPointCloud();
    void postProcess();

    // help functions
    template <typename T>
    void pointBodyToWorld(const Eigen::Matrix<T, 3, 1> &pi, Eigen::Matrix<T, 3, 1> &po)
    {
        V3D p_body(pi[0], pi[1], pi[2]);
        // offset_T_L_I: lidar和imu外参位移量
        // offset_R_L_I: lidar和imu外参旋转量
        V3D p_global(state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos);

        po[0] = p_global(0);
        po[1] = p_global(1);
        po[2] = p_global(2);
    }

    void pointBodyToWorld(PointType const *const pi, PointType *const po)
    {
        V3D p_body(pi->x, pi->y, pi->z);
        V3D p_global(state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos);

        po->x = p_global(0);
        po->y = p_global(1);
        po->z = p_global(2);
        po->intensity = pi->intensity;
    }

    void RGBpointBodyToWorld(PointType const *const pi, PointType *const po)
    {
        V3D p_body(pi->x, pi->y, pi->z);
        V3D p_global(state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos);

        po->x = p_global(0);
        po->y = p_global(1);
        po->z = p_global(2);
        po->intensity = pi->intensity;
    }

    void RGBpointBodyLidarToIMU(PointType const * const pi, PointType * const po)
    {
        V3D p_body_lidar(pi->x, pi->y, pi->z);
        V3D p_body_imu(state_point.offset_R_L_I*p_body_lidar + state_point.offset_T_L_I);

        po->x = p_body_imu(0);
        po->y = p_body_imu(1);
        po->z = p_body_imu(2);
        po->intensity = pi->intensity;
    }

    void points_cache_collect()
    {
        PointVector points_history;
        ikdtree.acquire_removed_points(points_history);
        // for (int i = 0; i < points_history.size(); i++) _featsArray->push_back(points_history[i]);
    }

    std::shared_ptr<Preprocess> p_pre{nullptr};
    std::shared_ptr<ImuProcess> p_imu{nullptr};

    ros::Subscriber sub_pcl;
    ros::Subscriber sub_imu;
    ros::Publisher pubLaserCloudFull;
    ros::Publisher pubLaserCloudFull_body;
    ros::Publisher pubLaserCloudEffect;
    ros::Publisher pubLaserCloudMap;
    ros::Publisher pubOdomAftMapped;
    ros::Publisher pubPath;
    std::ofstream fout_pre, fout_out, fout_dbg;

    bool lidar_pushed{false};
    bool flg_first_scan{true};
    bool flg_exit = false;
    bool flg_EKF_inited;
    double lidar_mean_scantime{0.0};
    int scan_num{0};
    int kdtree_size_st = 0;
    int kdtree_size_end = 0;
    int add_point_size = 0;
    int kdtree_delete_counter = 0;
    BoxPointType LocalMap_Points;
    bool Localmap_Initialized = false;

    string root_dir = ROOT_DIR;
    mutex mtx_buffer;
    condition_variable sig_buffer;
    double last_timestamp_lidar{0};
    double timediff_lidar_wrt_imu = 0.0;
    bool timediff_set_flg = false;

    bool path_en = true;
    bool scan_pub_en = false;
    bool dense_pub_en = false;
    bool scan_body_pub_en = false;
    int iterCount = 0, feats_down_size = 0;
    int NUM_MAX_ITERATIONS = 0;
    int laserCloudValidNum = 0;
    int pcd_save_interval = -1;
    int pcd_index = 0;

    bool runtime_pos_log = false;
    bool pcd_save_en = false;
    bool time_sync_en = false;
    bool extrinsic_est_en = true;
    std::string map_file_path;
    std::string lid_topic;
    std::string imu_topic;
    double filter_size_corner_min = 0;
    double filter_size_surf_min = 0;
    double filter_size_map_min = 0;
    double fov_deg = 0;
    double cube_len = 0;
    double HALF_FOV_COS = 0;
    double FOV_DEG = 0;
    double total_distance = 0;
    double lidar_end_time = 0;
    double first_lidar_time = 0.0;
    float DET_RANGE = 300.0f;
    double last_timestamp_imu = -1.0;
    double gyr_cov = 0.1;
    double acc_cov = 0.1;
    double b_gyr_cov = 0.0001;
    double b_acc_cov = 0.0001;
    int effct_feat_num = 0;
    int time_log_counter = 0;
    int scan_count = 0;
    int publish_count = 0;

    double res_mean_last = 0.05;
    double total_residual = 0.0;
    double T1[MAXN];
    double s_plot[MAXN];
    double s_plot2[MAXN];
    double s_plot3[MAXN];
    double s_plot4[MAXN];
    double s_plot5[MAXN];
    double s_plot6[MAXN];
    double s_plot7[MAXN];
    double s_plot8[MAXN];
    double s_plot9[MAXN];
    double s_plot10[MAXN];
    double s_plot11[MAXN];

    double kdtree_incremental_time = 0.0;
    double kdtree_search_time = 0.0;
    double kdtree_delete_time = 0.0;

    double match_time = 0;
    double solve_time = 0;
    double solve_const_H_time = 0;

    // nav_msgs::Path path;
    nav_msgs::Odometry odomAftMapped;
    geometry_msgs::Quaternion geoQuat;
    geometry_msgs::PoseStamped msg_body_pose;

    bool point_selected_surf[100000]{0};
    float res_last[100000]{0.0};
    const float MOV_THRESHOLD = 1.5f;
    double time_diff_lidar_to_imu = 0.0;

    V3F XAxisPoint_body{LIDAR_SP_LEN, 0.0, 0.0};
    V3F XAxisPoint_world{LIDAR_SP_LEN, 0.0, 0.0};
    V3D euler_cur;
    V3D position_last{Zero3d};
    V3D Lidar_T_wrt_IMU{Zero3d};
    M3D Lidar_R_wrt_IMU{Eye3d};

    /*** EKF inputs and output ***/
    MeasureGroup measures;
    esekfom::esekf<state_ikfom, 12, input_ikfom> kf;
    state_ikfom state_point;
    state_ikfom last_saved_state_point;
    nav_msgs::Path path_; // path to store the trajectory
    vect3 pos_lid;

    vector<vector<int>> pointSearchInd_surf;
    vector<BoxPointType> cub_needrm;
    vector<PointVector> Nearest_Points;
    vector<double> extrinT;
    vector<double> extrinR;
    deque<double> time_buffer;
    deque<PointCloudXYZI::Ptr> lidar_buffer;
    deque<sensor_msgs::Imu::ConstPtr> imu_buffer;

    PointCloudXYZI::Ptr featsFromMap{new PointCloudXYZI()};
    PointCloudXYZI::Ptr feats_undistort{new PointCloudXYZI()};
    PointCloudXYZI::Ptr feats_down_body{new PointCloudXYZI()};
    PointCloudXYZI::Ptr feats_down_world{new PointCloudXYZI()};
    PointCloudXYZI::Ptr normvec{new PointCloudXYZI(100000, 1)};
    PointCloudXYZI::Ptr laserCloudOri{new PointCloudXYZI(100000, 1)};
    PointCloudXYZI::Ptr corr_normvect{new PointCloudXYZI(100000, 1)};
    PointCloudXYZI::Ptr _featsArray;
    PointCloudXYZI::Ptr pcl_wait_pub{new PointCloudXYZI(500000, 1)};
    PointCloudXYZI::Ptr pcl_wait_save{new PointCloudXYZI()};

    pcl::VoxelGrid<PointType> downSizeFilterSurf;
    pcl::VoxelGrid<PointType> downSizeFilterMap;

    KD_TREE<PointType> ikdtree;

    pcl::VoxelGrid<pcl::PointXYZINormal> save_pcd_filter;
    ofstream pose_fs; // save pose to BAML pose file, https://github.com/hku-mars/BALM/issues/27#issuecomment-1259446844
    std::string baml_file_dir;
    std::string baml_folder_prefix;
    std::string baml_file_dir_pcd; // hba pcd file dir
    double distance_threshold=0.5;
};
