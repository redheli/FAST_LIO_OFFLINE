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

#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <fstream>
#include <csignal>
#include <unistd.h>
#include <Python.h>
#include <so3_math.h>
#include <ros/ros.h>
#include <iostream> 
#include <yaml-cpp/yaml.h>

#include "Mapping.hpp"
#include "Utils.hpp"

LaserMapping::LaserMapping()
    : extrinT(3, 0.0),
      extrinR(9, 0.0)
{
    p_pre = std::make_shared<Preprocess>();
    p_imu = std::make_shared<ImuProcess>();

    // create folder under ./PCD with current time
    time_t t = time(0);
    char tmp[64];
    strftime(tmp, sizeof(tmp), "%Y%m%d_%H%M%S", localtime(&t));
    auto baml_file_dir = std::string(ROOT_DIR) + "/PCD/" + std::string(tmp);
    std::cout<<"Pose File Dir: "<<baml_file_dir<<std::endl;
    mkdir(baml_file_dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    baml_pose_fs.open(baml_file_dir + "/alidarPose.csv", std::ios::out);
    baml_pose_fs.precision(6);
    baml_pose_fs<<std::fixed;
}

void LaserMapping::initOthers()
{
    ROS_INFO("LaserMapping initOthers");

    FOV_DEG = (fov_deg + 10.0) > 179.9 ? 179.9 : (fov_deg + 10.0);
    HALF_FOV_COS = cos((FOV_DEG) * 0.5 * PI_M / 180.0);

    _featsArray.reset(new PointCloudXYZI());

    memset(point_selected_surf, true, sizeof(point_selected_surf));
    memset(res_last, -1000.0f, sizeof(res_last));
    downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
    downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);
    memset(point_selected_surf, true, sizeof(point_selected_surf));
    memset(res_last, -1000.0f, sizeof(res_last));

    Lidar_T_wrt_IMU << VEC_FROM_ARRAY(extrinT);
    Lidar_R_wrt_IMU << MAT_FROM_ARRAY(extrinR);
    p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
    p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
    p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));
    p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
    p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));

    double epsi[23] = {0.001};
    fill(epsi, epsi + 23, 0.001);
    kf.init_dyn_share(
        get_f,
        df_dx,
        df_dw,
        [this](state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
        { h_share_model(s, ekfom_data); },
        NUM_MAX_ITERATIONS, epsi);

    /*** debug record ***/
    FILE *fp;
    string pos_log_dir = root_dir + "/Log/pos_log.txt";
    fp = fopen(pos_log_dir.c_str(), "w");

    // ofstream fout_pre, fout_out, fout_dbg;
    fout_pre.open(DEBUG_FILE_DIR("mat_pre.txt"), ios::out);
    fout_out.open(DEBUG_FILE_DIR("mat_out.txt"), ios::out);
    fout_dbg.open(DEBUG_FILE_DIR("dbg.txt"), ios::out);
    if (fout_pre && fout_out)
        cout << "~~~~" << ROOT_DIR << " file opened" << endl;
    else
        cout << "~~~~" << ROOT_DIR << " doesn't exist" << endl;
}

void LaserMapping::initOnline(ros::NodeHandle &nh)
{
    ROS_INFO("LaserMapping initOnline");
    nh.param<bool>("publish/path_en", path_en, true);
    nh.param<bool>("publish/scan_publish_en", scan_pub_en, true);
    nh.param<bool>("publish/dense_publish_en", dense_pub_en, true);
    nh.param<bool>("publish/scan_bodyframe_pub_en", scan_body_pub_en, true);
    nh.param<int>("max_iteration", NUM_MAX_ITERATIONS, 4);
    nh.param<string>("map_file_path", map_file_path, "");
    nh.param<string>("common/lid_topic", lid_topic, "/livox/lidar");
    nh.param<string>("common/imu_topic", imu_topic, "/livox/imu");
    nh.param<bool>("common/time_sync_en", time_sync_en, false);
    nh.param<double>("common/time_offset_lidar_to_imu", time_diff_lidar_to_imu, 0.0);
    nh.param<double>("filter_size_corner", filter_size_corner_min, 0.5);
    nh.param<double>("filter_size_surf", filter_size_surf_min, 0.5);
    nh.param<double>("filter_size_map", filter_size_map_min, 0.5);
    nh.param<double>("cube_side_length", cube_len, 200);
    nh.param<float>("mapping/det_range", DET_RANGE, 300.f);
    nh.param<double>("mapping/fov_degree", fov_deg, 180);
    nh.param<double>("mapping/gyr_cov", gyr_cov, 0.1);
    nh.param<double>("mapping/acc_cov", acc_cov, 0.1);
    nh.param<double>("mapping/b_gyr_cov", b_gyr_cov, 0.0001);
    nh.param<double>("mapping/b_acc_cov", b_acc_cov, 0.0001);
    nh.param<double>("preprocess/blind", p_pre->blind, 0.01);
    nh.param<int>("preprocess/lidar_type", p_pre->lidar_type, AVIA);
    nh.param<int>("preprocess/scan_line", p_pre->N_SCANS, 16);
    nh.param<int>("preprocess/timestamp_unit", p_pre->time_unit, US);
    nh.param<int>("preprocess/scan_rate", p_pre->SCAN_RATE, 10);
    nh.param<int>("point_filter_num", p_pre->point_filter_num, 2);
    nh.param<bool>("feature_extract_enable", p_pre->feature_enabled, false);
    nh.param<bool>("runtime_pos_log_enable", runtime_pos_log, 0);
    nh.param<bool>("mapping/extrinsic_est_en", extrinsic_est_en, true);
    nh.param<bool>("pcd_save/pcd_save_en", pcd_save_en, false);
    nh.param<int>("pcd_save/interval", pcd_save_interval, -1);
    nh.param<std::vector<double>>("mapping/extrinsic_T", extrinT, std::vector<double>());
    nh.param<std::vector<double>>("mapping/extrinsic_R", extrinR, std::vector<double>());
    ROS_INFO("LaserMapping initOnline2");
    if (p_pre->lidar_type == AVIA)
    {
        sub_pcl = nh.subscribe<livox_ros_driver::CustomMsg>(lid_topic, 200000,
                                                            [this](const livox_ros_driver::CustomMsg::ConstPtr &msg)
                                                            { livox_pcl_cbk(msg); });
    }
    else
    {
        sub_pcl = nh.subscribe<sensor_msgs::PointCloud2>(lid_topic, 200000,
                                                         [this](const sensor_msgs::PointCloud2::ConstPtr &msg)
                                                         { standard_pcl_cbk(msg); });
    }

    sub_imu = nh.subscribe<sensor_msgs::Imu>(imu_topic, 200000, [this](const sensor_msgs::Imu::ConstPtr &msg)
                                             { imu_cbk(msg); });

    pubLaserCloudFull = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100000);
    pubLaserCloudFull_body = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered_body", 100000);
    pubLaserCloudEffect = nh.advertise<sensor_msgs::PointCloud2>("/cloud_effected", 100000);
    pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>("/Laser_map", 100000);
    pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/Odometry", 100000);
    pubPath = nh.advertise<nav_msgs::Path>("/path", 100000);

    initOthers();
}

void LaserMapping::initWithoutROS(const std::string &config_yaml)
{
    ROS_INFO_STREAM("config file: " << config_yaml);
    LoadParamsFromYAML(config_yaml);
    initOthers();
}

void LaserMapping::LoadParamsFromYAML(const std::string &yaml_file)
{
    auto yaml = YAML::LoadFile(yaml_file);
    ROS_INFO_STREAM("[LoadParamsFromYAML] yaml file: " << yaml_file);
    path_en = yaml["publish"]["path_en"].as<bool>(true);
    ROS_INFO_STREAM("path_en: " << path_en);

    scan_pub_en = yaml["publish"]["scan_publish_en"].as<bool>(true);
    ROS_INFO_STREAM("scan_pub_en: " << scan_pub_en);

    dense_pub_en = yaml["publish"]["dense_publish_en"].as<bool>(true);
    ROS_INFO_STREAM("dense_pub_en: " << dense_pub_en);

    scan_body_pub_en = yaml["publish"]["scan_bodyframe_pub_en"].as<bool>(true);
    ROS_INFO_STREAM("scan_body_pub_en: " << scan_body_pub_en);

    NUM_MAX_ITERATIONS = yaml["common"]["max_iteration"].as<int>(4);
    ROS_INFO_STREAM("NUM_MAX_ITERATIONS: " << NUM_MAX_ITERATIONS);

    map_file_path = yaml["map_file_path"].as<std::string>("");
    ROS_INFO_STREAM("map_file_path: " << map_file_path);

    // nh.param<string>("common/lid_topic", lid_topic, "/livox/lidar");
    lid_topic = yaml["common"]["lid_topic"].as<std::string>("/livox/lidar");
    ROS_INFO_STREAM("lid_topic: " << lid_topic);

    // nh.param<string>("common/imu_topic", imu_topic, "/livox/imu");
    imu_topic = yaml["common"]["imu_topic"].as<std::string>("/livox/imu");
    ROS_INFO_STREAM("imu_topic: " << imu_topic);

    // nh.param<bool>("common/time_sync_en", time_sync_en, false);
    time_sync_en = yaml["common"]["time_sync_en"].as<bool>(false);
    ROS_INFO_STREAM("time_sync_en: " << time_sync_en);

    // nh.param<double>("common/time_offset_lidar_to_imu", time_diff_lidar_to_imu, 0.0);
    time_diff_lidar_to_imu = yaml["common"]["time_offset_lidar_to_imu"].as<double>(0.0);
    ROS_INFO_STREAM("time_diff_lidar_to_imu: " << time_diff_lidar_to_imu);

    // nh.param<double>("filter_size_corner", filter_size_corner_min, 0.5);
    filter_size_corner_min = yaml["common"]["filter_size_corner"].as<double>(0.5);
    ROS_INFO_STREAM("filter_size_corner_min: " << filter_size_corner_min);

    // nh.param<double>("filter_size_surf", filter_size_surf_min, 0.5);
    filter_size_surf_min = yaml["common"]["filter_size_surf"].as<double>(0.5);
    ROS_INFO_STREAM("filter_size_surf_min: " << filter_size_surf_min);

    // nh.param<double>("filter_size_map", filter_size_map_min, 0.5);
    filter_size_map_min = yaml["common"]["filter_size_map"].as<double>(0.5);
    ROS_INFO_STREAM("filter_size_map_min: " << filter_size_map_min);

    // nh.param<double>("cube_side_length", cube_len, 200);
    cube_len = yaml["common"]["cube_side_length"].as<double>(200);
    ROS_INFO_STREAM("cube_len: " << cube_len);

    // nh.param<float>("mapping/det_range", DET_RANGE, 300.f);
    DET_RANGE = yaml["mapping"]["det_range"].as<float>(300.f);
    ROS_INFO_STREAM("DET_RANGE: " << DET_RANGE);

    // nh.param<double>("mapping/fov_degree", fov_deg, 180);
    fov_deg = yaml["mapping"]["fov_degree"].as<double>(180);
    ROS_INFO_STREAM("fov_deg: " << fov_deg);

    // nh.param<double>("mapping/gyr_cov", gyr_cov, 0.1);
    gyr_cov = yaml["mapping"]["gyr_cov"].as<double>(0.1);
    ROS_INFO_STREAM("gyr_cov: " << gyr_cov);

    // nh.param<double>("mapping/acc_cov", acc_cov, 0.1);
    acc_cov = yaml["mapping"]["acc_cov"].as<double>(0.1);
    ROS_INFO_STREAM("acc_cov: " << acc_cov);

    // nh.param<double>("mapping/b_gyr_cov", b_gyr_cov, 0.0001);
    b_gyr_cov = yaml["mapping"]["b_gyr_cov"].as<double>(0.0001);
    ROS_INFO_STREAM("b_gyr_cov: " << b_gyr_cov);

    // nh.param<double>("mapping/b_acc_cov", b_acc_cov, 0.0001);
    b_acc_cov = yaml["mapping"]["b_acc_cov"].as<double>(0.0001);
    ROS_INFO_STREAM("b_acc_cov: " << b_acc_cov);

    // nh.param<double>("preprocess/blind", p_pre->blind, 0.01);
    p_pre->blind = yaml["preprocess"]["blind"].as<double>(0.01);
    ROS_INFO_STREAM("p_pre->blind: " << p_pre->blind);

    // nh.param<int>("preprocess/lidar_type", p_pre->lidar_type, AVIA);
    p_pre->lidar_type = yaml["preprocess"]["lidar_type"].as<int>(AVIA);
    ROS_INFO_STREAM("p_pre->lidar_type: " << p_pre->lidar_type);

    // nh.param<int>("preprocess/scan_line", p_pre->N_SCANS, 16);
    p_pre->N_SCANS = yaml["preprocess"]["scan_line"].as<int>(16);
    ROS_INFO_STREAM("p_pre->N_SCANS: " << p_pre->N_SCANS);

    // nh.param<int>("preprocess/timestamp_unit", p_pre->time_unit, US);
    p_pre->time_unit = yaml["preprocess"]["timestamp_unit"].as<int>(US);
    ROS_INFO_STREAM("p_pre->time_unit: " << p_pre->time_unit);

    // nh.param<int>("preprocess/scan_rate", p_pre->SCAN_RATE, 10);
    p_pre->SCAN_RATE = yaml["preprocess"]["scan_rate"].as<int>(10);
    ROS_INFO_STREAM("p_pre->SCAN_RATE: " << p_pre->SCAN_RATE);

    // nh.param<int>("point_filter_num", p_pre->point_filter_num, 2);
    p_pre->point_filter_num = yaml["common"]["point_filter_num"].as<int>(2);
    ROS_INFO_STREAM("p_pre->point_filter_num: " << p_pre->point_filter_num);

    // nh.param<bool>("feature_extract_enable", p_pre->feature_enabled, false);
    p_pre->feature_enabled = yaml["common"]["feature_extract_enable"].as<bool>(false);
    ROS_INFO_STREAM("p_pre->feature_enabled: " << p_pre->feature_enabled);

    // nh.param<bool>("runtime_pos_log_enable", runtime_pos_log, 0);
    runtime_pos_log = yaml["common"]["runtime_pos_log_enable"].as<bool>(0);
    ROS_INFO_STREAM("runtime_pos_log: " << runtime_pos_log);

    // nh.param<bool>("mapping/extrinsic_est_en", extrinsic_est_en, true);
    extrinsic_est_en = yaml["mapping"]["extrinsic_est_en"].as<bool>(true);
    ROS_INFO_STREAM("extrinsic_est_en: " << extrinsic_est_en);

    // nh.param<bool>("pcd_save/pcd_save_en", pcd_save_en, false);
    pcd_save_en = yaml["pcd_save"]["pcd_save_en"].as<bool>(false);
    ROS_INFO_STREAM("pcd_save_en: " << pcd_save_en);

    // nh.param<int>("pcd_save/interval", pcd_save_interval, -1);
    pcd_save_interval = yaml["pcd_save"]["interval"].as<int>(-1);
    ROS_INFO_STREAM("pcd_save_interval: " << pcd_save_interval);

    // nh.param<std::vector<double>>("mapping/extrinsic_T", extrinT, std::vector<double>());
    extrinT = yaml["mapping"]["extrinsic_T"].as<std::vector<double>>(std::vector<double>());
    ROS_INFO_STREAM("extrinT: " << extrinT[0] << " " << extrinT[1] << " " << extrinT[2]);

    // nh.param<std::vector<double>>("mapping/extrinsic_R", extrinR, std::vector<double>());
    extrinR = yaml["mapping"]["extrinsic_R"].as<std::vector<double>>(std::vector<double>());
    ROS_INFO_STREAM("extrinR: " << extrinR[0] << " " << extrinR[1] << " " << extrinR[2] << " " << extrinR[3] << " " << extrinR[4] << " " << extrinR[5] << " " << extrinR[6] << " " << extrinR[7] << " " << extrinR[8]);
    ROS_INFO_STREAM("LoadParamsFromYAML done");
}

void LaserMapping::RunOnce()
{
    std::lock_guard<std::mutex> lock(mtx_buffer);
    if (!sync_packages())
    {
        return;
    }

    if (flg_first_scan)
    {
        first_lidar_time = measures.lidar_beg_time;
        p_imu->first_lidar_time = first_lidar_time;
        flg_first_scan = false;
        std::cout << "debug 0" << std::endl;
        return;
    }

    double t0, t1, t2, t3, t4, t5, match_start, solve_start, svd_time;

    match_time = 0;
    double kdtree_search_time = 0.0;
    solve_time = 0;
    solve_const_H_time = 0;
    svd_time = 0;
    t0 = omp_get_wtime();

    p_imu->Process(measures, kf, feats_undistort);
    double imu_lidar_diff = measures.lidar_beg_time - measures.imu.back()->header.stamp.toSec();
    
    state_point = kf.get_x();
    pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
    if (feats_undistort->empty() || (feats_undistort == NULL))
    {
        ROS_WARN("No point, skip this scan!\n");
        return;
    }

    flg_EKF_inited = (measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? false : true;
    /*** Segment the map in lidar FOV ***/
    // 动态调整局部地图,在拿到eskf前馈结果后
    lasermap_fov_segment();

    /*** downsample the feature points in a scan ***/
    downSizeFilterSurf.setInputCloud(feats_undistort); // 获得去畸变后的点云数据
    downSizeFilterSurf.filter(*feats_down_body);       // 滤波降采样后的点云数据
    t1 = omp_get_wtime();                              // 记录时间
    feats_down_size = feats_down_body->points.size();  // 记录滤波后的点云数量
    /*** initialize the map kdtree ***/
    // 构建kd树
    if (ikdtree.Root_Node == nullptr)
    {
        if (feats_down_size > 5)
        {
            // 设置ikd tree的降采样参数
            ikdtree.set_downsample_param(filter_size_map_min);
            feats_down_world->resize(feats_down_size); // 将下采样得到的地图点大小于body系大小一致
            for (int i = 0; i < feats_down_size; i++)
            {
                pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i])); // 将下采样得到的地图点转换为世界坐标系下的点云
            }
            // 组织ikd tree
            ikdtree.Build(feats_down_world->points);
        }
        ROS_WARN("Init ikdtree!\n");
        return;
    }
    // 获取ikd tree中的有效节点数，无效点就是被打了deleted标签的点
    int featsFromMapNum = ikdtree.validnum();
    // 获取Ikd tree中的节点数
    kdtree_size_st = ikdtree.size();


    /*** ICP and iterated Kalman filter update ***/
    if (feats_down_size < 5)
    {
        ROS_WARN("No point, skip this scan!\n");
        return;
    }

    // ICP和迭代卡尔曼滤波更新
    normvec->resize(feats_down_size);
    feats_down_world->resize(feats_down_size);

    // 外参，旋转矩阵转欧拉角
    V3D ext_euler = SO3ToEuler(state_point.offset_R_L_I);
    fout_pre << setw(20) << measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << state_point.pos.transpose() << " " << ext_euler.transpose() << " " << state_point.offset_T_L_I.transpose() << " " << state_point.vel.transpose()
             << " " << state_point.bg.transpose() << " " << state_point.ba.transpose() << " " << state_point.grav << endl;
    if (0) // If you need to see map point, change to "if(1)"
    {
        // 释放PCL_Storage的内存
        PointVector().swap(ikdtree.PCL_Storage);
        // 把树展平用于展示
        ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);
        featsFromMap->clear();
        featsFromMap->points = ikdtree.PCL_Storage;
    }

    pointSearchInd_surf.resize(feats_down_size); // 搜索索引
    Nearest_Points.resize(feats_down_size);      // 将降采样处理后的点云用于搜索最近点
    int rematch_num = 0;
    bool nearest_search_en = true; //

    t2 = omp_get_wtime();

    /*** iterated state estimation ***/
    /*** 迭代状态估计 ***/
    double t_update_start = omp_get_wtime();
    double solve_H_time = 0;
    // 迭代卡尔曼滤波更新，更新地图信息
    kf.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time);
    state_point = kf.get_x();
    euler_cur = SO3ToEuler(state_point.rot);
    pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
    geoQuat.x = state_point.rot.coeffs()[0];
    geoQuat.y = state_point.rot.coeffs()[1];
    geoQuat.z = state_point.rot.coeffs()[2];
    geoQuat.w = state_point.rot.coeffs()[3];

    double t_update_end = omp_get_wtime();

    /******* Publish odometry *******/
    // publish_odometry(pubOdomAftMapped);

    /*** add the feature points to map kdtree ***/
    t3 = omp_get_wtime();
    map_incremental();
    t5 = omp_get_wtime();

    // save pose to path
    geometry_msgs::PoseStamped body_pose;
    body_pose.pose.position.x = state_point.pos(0);
    body_pose.pose.position.y = state_point.pos(1);
    body_pose.pose.position.z = state_point.pos(2);
    body_pose.pose.orientation.x = geoQuat.x;
    body_pose.pose.orientation.y = geoQuat.y;
    body_pose.pose.orientation.z = geoQuat.z;
    body_pose.pose.orientation.w = geoQuat.w;
    ros::Time lidar_timestamp{0};
    lidar_timestamp.fromSec(measures.lidar_beg_time);
    body_pose.header.stamp = lidar_timestamp;
    path_.poses.push_back(body_pose);

    ROS_INFO_STREAM("[RunOnce] pose xyz: " << state_point.pos.transpose());

    // TODO save pose to file

    {
        publish_frame_world(pubLaserCloudFull);
        savePoseAndPointCloud();
    }
    
}

void LaserMapping::savePCD()
{
    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. pcd save will largely influence the real-time performences **/
    // if (pcl_wait_save->size() > 0 && pcd_save_en)
    {
        string file_name = string("scans_offline.pcd");
        string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
        pcl::PCDWriter pcd_writer;
        cout << "Writing to pcd ......" << file_name << endl;
        pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
        cout << "....... Done" << file_name << endl;
        cout << "current scan saved to /PCD/" << file_name << endl;
    }
}

void LaserMapping::savePoseAndPointCloud()
{
    // save pose to file
    Eigen::Matrix4d outT;
    outT << state_point.rot.toRotationMatrix(), state_point.pos, 0, 0, 0, measures.lidar_beg_time;
    for (int j = 0; j < 4; j++)
    {
        for (int k = 0; k < 4; k++)
        baml_pose_fs << outT(j, k) << ",";
        baml_pose_fs << endl;
    }
    // string pose_file_name = string("pose.txt");
    // string pose_dir(string(string(ROOT_DIR) + "PCD/") + pose_file_name);
    // ofstream fout_pose;
    // fout_pose.open(pose_dir, ios::out);
    // if (fout_pose)
    // {
    //     fout_pose << "pose xyz: " << state_point.pos.transpose() << endl;
    //     fout_pose << "pose quat: " << geoQuat.x << " " << geoQuat.y << " " << geoQuat.z << " " << geoQuat.w << endl;
    //     fout_pose.close();
    // }
    // else
    // {
    //     cout << "pose file open failed" << endl;
    // }

    // save pointcloud to file
    string pointcloud_file_name = string("pointcloud.txt");
    string pointcloud_dir(string(string(ROOT_DIR) + "PCD/") + pointcloud_file_name);
    ofstream fout_pointcloud;
    fout_pointcloud.open(pointcloud_dir, ios::out);
    if (fout_pointcloud)
    {
        fout_pointcloud << "pointcloud size: " << feats_undistort->size() << endl;
        for (int i = 0; i < feats_undistort->size(); i++)
        {
            fout_pointcloud << feats_undistort->points[i].x << " " << feats_undistort->points[i].y << " " << feats_undistort->points[i].z << endl;
        }
        fout_pointcloud.close();
    }
    else
    {
        cout << "pointcloud file open failed" << endl;
    }
}

void LaserMapping::publish_frame_world(const ros::Publisher &pubLaserCloudFull)
{
    if (scan_pub_en) // 设置是否发布激光雷达数据，是否发布稠密数据，是否发布激光雷达数据的身体数据
    {
        PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body); // 判断是否需要降采样
        int size = laserCloudFullRes->points.size();                                             // 获取待转换点云的大小
        PointCloudXYZI::Ptr laserCloudWorld(
            new PointCloudXYZI(size, 1)); // 创建一个点云用于存储转换到世界坐标系的点云

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&laserCloudFullRes->points[i],
                                &laserCloudWorld->points[i]);
        }

        sensor_msgs::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
        laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
        laserCloudmsg.header.frame_id = "camera_init";
        pubLaserCloudFull.publish(laserCloudmsg);
        publish_count -= PUBFRAME_PERIOD;
    }

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. noted that pcd save will influence the real-time performences **/
    // if (pcd_save_en)
    {
        int size = feats_undistort->points.size();
        PointCloudXYZI::Ptr laserCloudWorld(
            new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&feats_undistort->points[i],
                                &laserCloudWorld->points[i]);
        }

        // downsample
        // PointCloudXYZI::Ptr cloud_filtered(new PointCloudXYZI());

        // save_pcd_filter.setInputCloud(laserCloudWorld);
        // save_pcd_filter.setLeafSize(0.1f, 0.1f, 0.1f);
        // save_pcd_filter.filter(*cloud_filtered);

        *pcl_wait_save += *laserCloudWorld;
        ROS_INFO_STREAM("pcl_wait_save size: " << pcl_wait_save->size());

        static int scan_wait_num = 0;
        scan_wait_num++;
        if (pcl_wait_save->size() > 0 && pcd_save_interval > 0 && scan_wait_num >= pcd_save_interval)
        {
            pcd_index++;
            string all_points_dir(string(string(ROOT_DIR) + "PCD/scans_") + to_string(pcd_index) + string(".pcd"));
            pcl::PCDWriter pcd_writer;
            cout << "current scan saved to /PCD/" << all_points_dir << endl;
            pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
            pcl_wait_save->clear();
            scan_wait_num = 0;
        }
    }
}

void LaserMapping::map_incremental()
{
    PointVector PointToAdd;
    PointVector PointNoNeedDownsample;
    PointToAdd.reserve(feats_down_size);
    PointNoNeedDownsample.reserve(feats_down_size);
    for (int i = 0; i < feats_down_size; i++)
    {
        /* transform to world frame */
        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
        /* decide if need add to map */
        if (!Nearest_Points[i].empty() && flg_EKF_inited)
        {
            const PointVector &points_near = Nearest_Points[i];
            bool need_add = true;
            BoxPointType Box_of_Point;
            PointType downsample_result, mid_point;
            mid_point.x = floor(feats_down_world->points[i].x / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.y = floor(feats_down_world->points[i].y / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.z = floor(feats_down_world->points[i].z / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
            float dist = calc_dist(feats_down_world->points[i], mid_point);
            if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_map_min && fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_map_min && fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_map_min)
            {
                PointNoNeedDownsample.push_back(feats_down_world->points[i]);
                continue;
            }
            for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i++)
            {
                if (points_near.size() < NUM_MATCH_POINTS)
                    break;
                if (calc_dist(points_near[readd_i], mid_point) < dist)
                {
                    need_add = false;
                    break;
                }
            }
            if (need_add)
                PointToAdd.push_back(feats_down_world->points[i]);
        }
        else
        {
            PointToAdd.push_back(feats_down_world->points[i]);
        }
    }

    double st_time = omp_get_wtime();
    add_point_size = ikdtree.Add_Points(PointToAdd, true);
    ikdtree.Add_Points(PointNoNeedDownsample, false);
    add_point_size = PointToAdd.size() + PointNoNeedDownsample.size();
    kdtree_incremental_time = omp_get_wtime() - st_time;
}

void LaserMapping::lasermap_fov_segment()
{
    cub_needrm.clear();
    kdtree_delete_counter = 0;
    kdtree_delete_time = 0.0;
    pointBodyToWorld(XAxisPoint_body, XAxisPoint_world);
    V3D pos_LiD = pos_lid;
    if (!Localmap_Initialized)
    {
        for (int i = 0; i < 3; i++)
        {
            LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
            LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
        }
        Localmap_Initialized = true;
        return;
    }
    float dist_to_map_edge[3][2];
    bool need_move = false;
    for (int i = 0; i < 3; i++)
    {
        dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);
        dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE)
            need_move = true;
    }
    if (!need_move)
        return;
    BoxPointType New_LocalMap_Points, tmp_boxpoints;
    New_LocalMap_Points = LocalMap_Points;
    float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9, double(DET_RANGE * (MOV_THRESHOLD - 1)));
    for (int i = 0; i < 3; i++)
    {
        tmp_boxpoints = LocalMap_Points;
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE)
        {
            New_LocalMap_Points.vertex_max[i] -= mov_dist;
            New_LocalMap_Points.vertex_min[i] -= mov_dist;
            tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        }
        else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE)
        {
            New_LocalMap_Points.vertex_max[i] += mov_dist;
            New_LocalMap_Points.vertex_min[i] += mov_dist;
            tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        }
    }
    LocalMap_Points = New_LocalMap_Points;

    points_cache_collect();
    double delete_begin = omp_get_wtime();
    if (cub_needrm.size() > 0)
        kdtree_delete_counter = ikdtree.Delete_Point_Boxes(cub_needrm);
    kdtree_delete_time = omp_get_wtime() - delete_begin;
}

bool LaserMapping::sync_packages()
{
    if (lidar_buffer.empty() || imu_buffer.empty())
    {
        return false;
    }

    /*** push a lidar scan ***/
    if (!lidar_pushed)
    {
        measures.lidar = lidar_buffer.front();
        measures.lidar_beg_time = time_buffer.front();
        if (measures.lidar->points.size() <= 1) // time too little
        {
            lidar_end_time = measures.lidar_beg_time + lidar_mean_scantime;
            ROS_WARN("Too few input point cloud!\n");
        }
        else if (measures.lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime)
        {
            lidar_end_time = measures.lidar_beg_time + lidar_mean_scantime;
        }
        else
        {
            scan_num++;
            lidar_end_time = measures.lidar_beg_time + measures.lidar->points.back().curvature / double(1000);
            lidar_mean_scantime += (measures.lidar->points.back().curvature / double(1000) - lidar_mean_scantime) / scan_num;
        }

        measures.lidar_end_time = lidar_end_time;

        lidar_pushed = true;
    }

    if (last_timestamp_imu < lidar_end_time)
    {
        return false;
    }

    /*** push imu data, and pop from imu buffer ***/
    double imu_time = imu_buffer.front()->header.stamp.toSec();
    measures.imu.clear();
    while ((!imu_buffer.empty()) && (imu_time < lidar_end_time))
    {
        imu_time = imu_buffer.front()->header.stamp.toSec();
        if (imu_time > lidar_end_time)
            break;
        measures.imu.push_back(imu_buffer.front());
        imu_buffer.pop_front();
    }

    lidar_buffer.pop_front();
    time_buffer.pop_front();
    lidar_pushed = false;
    return true;
}

void LaserMapping::imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in)
{
    publish_count++;
    static int imu_cbk_count = 0;
    sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));

    if (abs(timediff_lidar_wrt_imu) > 0.1 && time_sync_en)
    {
        msg->header.stamp =
            ros::Time().fromSec(timediff_lidar_wrt_imu + msg_in->header.stamp.toSec());
    }

    msg->header.stamp = ros::Time().fromSec(msg_in->header.stamp.toSec() - time_diff_lidar_to_imu);

    double timestamp = msg->header.stamp.toSec();

    mtx_buffer.lock();

    if (timestamp < last_timestamp_imu)
    {
        ROS_WARN("imu loop back, clear buffer");
        throw std::runtime_error("imu loop back, clear buffer");
        imu_buffer.clear();
    }

    last_timestamp_imu = timestamp;

    imu_buffer.push_back(msg);
    imu_cbk_count++;
    mtx_buffer.unlock();
}

void LaserMapping::standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
    mtx_buffer.lock();
    scan_count++;
    double preprocess_start_time = omp_get_wtime();
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        throw std::runtime_error("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }

    PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(msg->header.stamp.toSec());
    last_timestamp_lidar = msg->header.stamp.toSec();
    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
}

void LaserMapping::h_share_model(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
{
    double match_start = omp_get_wtime();
    laserCloudOri->clear();
    corr_normvect->clear();
    total_residual = 0.0;

/** closest surface search and residual computation **/
#ifdef MP_EN
    omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
    for (int i = 0; i < feats_down_size; i++)
    {
        PointType &point_body = feats_down_body->points[i];
        PointType &point_world = feats_down_world->points[i];

        /* transform to world frame */
        V3D p_body(point_body.x, point_body.y, point_body.z);
        V3D p_global(s.rot * (s.offset_R_L_I * p_body + s.offset_T_L_I) + s.pos);
        point_world.x = p_global(0);
        point_world.y = p_global(1);
        point_world.z = p_global(2);
        point_world.intensity = point_body.intensity;

        vector<float> pointSearchSqDis(NUM_MATCH_POINTS);

        auto &points_near = Nearest_Points[i];

        if (ekfom_data.converge)
        {
            /** Find the closest surfaces in the map **/
            ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
            point_selected_surf[i] = points_near.size() < NUM_MATCH_POINTS ? false : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false
                                                                                                                                : true;
        }

        if (!point_selected_surf[i])
            continue;

        VF(4)
        pabcd;
        point_selected_surf[i] = false;
        if (esti_plane(pabcd, points_near, 0.1f))
        {
            float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);
            float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());

            if (s > 0.9)
            {
                point_selected_surf[i] = true;
                normvec->points[i].x = pabcd(0);
                normvec->points[i].y = pabcd(1);
                normvec->points[i].z = pabcd(2);
                normvec->points[i].intensity = pd2;
                res_last[i] = abs(pd2);
            }
        }
    }

    effct_feat_num = 0;

    for (int i = 0; i < feats_down_size; i++)
    {
        if (point_selected_surf[i])
        {
            laserCloudOri->points[effct_feat_num] = feats_down_body->points[i];
            corr_normvect->points[effct_feat_num] = normvec->points[i];
            total_residual += res_last[i];
            effct_feat_num++;
        }
    }

    if (effct_feat_num < 1)
    {
        ekfom_data.valid = false;
        ROS_WARN("No Effective Points! \n");
        return;
    }

    res_mean_last = total_residual / effct_feat_num;
    match_time += omp_get_wtime() - match_start;
    double solve_start_ = omp_get_wtime();

    /*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/
    ekfom_data.h_x = MatrixXd::Zero(effct_feat_num, 12); // 23
    ekfom_data.h.resize(effct_feat_num);

    for (int i = 0; i < effct_feat_num; i++)
    {
        const PointType &laser_p = laserCloudOri->points[i];
        V3D point_this_be(laser_p.x, laser_p.y, laser_p.z);
        M3D point_be_crossmat;
        point_be_crossmat << SKEW_SYM_MATRX(point_this_be);
        V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I;
        M3D point_crossmat;
        point_crossmat << SKEW_SYM_MATRX(point_this);

        /*** get the normal vector of closest surface/corner ***/
        const PointType &norm_p = corr_normvect->points[i];
        V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

        /*** calculate the Measuremnt Jacobian matrix H ***/
        V3D C(s.rot.conjugate() * norm_vec);
        V3D A(point_crossmat * C);
        if (extrinsic_est_en)
        {
            V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C); // s.rot.conjugate()*norm_vec);
            ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
        }
        else
        {
            ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        }

        /*** Measuremnt: distance to the closest surface/corner ***/
        ekfom_data.h(i) = -norm_p.intensity;
    }
    solve_time += omp_get_wtime() - solve_start_;
}

void LaserMapping::livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg)
{
    mtx_buffer.lock();
    double preprocess_start_time = omp_get_wtime();
    scan_count++;
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        throw std::runtime_error("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }
    last_timestamp_lidar = msg->header.stamp.toSec();

    if (!time_sync_en && abs(last_timestamp_imu - last_timestamp_lidar) > 10.0 && !imu_buffer.empty() && !lidar_buffer.empty())
    {
        printf("IMU and LiDAR not Synced, IMU time: %lf, lidar header time: %lf \n", last_timestamp_imu, last_timestamp_lidar);
    }

    if (time_sync_en && !timediff_set_flg && abs(last_timestamp_lidar - last_timestamp_imu) > 1 && !imu_buffer.empty())
    {
        timediff_set_flg = true;
        timediff_lidar_wrt_imu = last_timestamp_lidar + 0.1 - last_timestamp_imu;
        printf("Self sync IMU and LiDAR, time diff is %.10lf \n", timediff_lidar_wrt_imu);
    }

    PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(last_timestamp_lidar);

    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
}