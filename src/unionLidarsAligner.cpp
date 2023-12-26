// The key objective of this code is to align the spinning lidar
// and livox code both on position and time side.
// and publish to a new topic, which the timestamp
// are same and under the same coordinate as livox_frame;
//  Qingqing Li @uTU

#include <time.h>
#include <ros/ros.h>
#include <tf/tf.h>
#include <tf/transform_broadcaster.h>
#include <tf_conversions/tf_eigen.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Imu.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/filter.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/search/kdtree.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/gicp.h>
#include "livox_ros_driver/CustomMsg.h"
#include "lidars_extrinsic_cali.h"
#include "preprocess.h"

#include <fstream>
#include <chrono>
#include <string>
#include <Eigen/Dense>
#include <mutex>
#include <stdint.h>

#include "sophus/so3.hpp"
#define PI 3.1415926
// #define BACKWARD_HAS_DW 1
// #include "backward.hpp"
// namespace backward {
// backward::SignalHandling sh;
// }
typedef uint64_t uint64;
typedef pcl::PointXYZINormal PointNormal;

class LidarsParamEstimator{
    private:
        ros::NodeHandle nh;
        // subscribe raw data
        ros::Subscriber sub_mid1;
        ros::Subscriber sub_mid2;
        ros::Subscriber sub_mid3;
        ros::Subscriber sub_velo;
        ros::Subscriber sub_ouster;
        ros::Subscriber sub_imu;

        // publish points shared FOV with Horizon
        ros::Publisher pub_velo_inFOV;

        // pub extrinsic aligned data
        ros::Publisher pub_hori;
        ros::Publisher pub_velo;
        ros::Publisher pub_hori_livoxmsg;
        // pub time_offset compensated data
        ros::Publisher pub_time_hori;
        ros::Publisher pub_time_velo;
        ros::Publisher pub_time_hori_livoxmsg;

        ros::Publisher pub_merged_cloud;

        // Hori TF
        int                         _hori_itegrate_frames = 10;
        int                         _hori_itegrate_frames2 = 10;
        int                         _hori_itegrate_frames3 = 10;
        pcl::PointCloud<PointType>  _hori_igcloud;
        pcl::PointCloud<PointType>  _hori_igcloud2;
        pcl::PointCloud<PointType>  _hori_igcloud3;
        Eigen::Matrix4f             _velo_hori_tf_matrix;
        Eigen::Matrix4f             _mid1_ouster_tf_init;
        Eigen::Matrix4f             _mid2_ouster_tf_init;
        Eigen::Matrix4f             _mid3_ouster_tf_init;
        Eigen::Matrix4f             _mid2_mid1_tf_init;
        Eigen::Matrix4f             _mid3_mid1_tf_init;
        bool                        _hori_tf_initd         = false;
        bool                        _first_velo_reveived= false;
        int                         _cut_raw_message_pieces = 2;

        // real time angular yaw speed
        double                      _yaw_velocity;

        pcl::PointCloud<PointType> _velo_new_cloud;
        pcl::PointCloud<PointType> _ouster_new_cloud;

        // raw message queue for time_offset
        // std::queue< sensor_msgs::PointCloud2 >          _velo_queue;
        std::vector<sensor_msgs::PointCloud2 >          _velo_queue;
        std::queue< pcl::PointCloud<pcl::PointXYZI> >   _velo_fov_queue;
        std::queue< livox_ros_driver::CustomMsg >       _hori_queue;
        std::queue< livox_ros_driver::CustomMsg >       _hori_queue2;
        std::vector<float>                              _hori_msg_yaw_vec;
        std::vector<sensor_msgs::Imu::ConstPtr>         _imu_vec;
        std::mutex _mutexIMUVec;
        std::mutex _mutexHoriQueue;
        std::mutex _mutexVeloQueue;

        uint64                                      _hori_start_stamp ;
        uint64                                      _hori_start_stamp2 ;
        uint64                                      _hori_start_stamp3 ;
        bool                                        _first_hori = true;
        bool                                        _first_hori2 = true;
        bool                                        _first_hori3 = true;
        std::mutex _mutexHoriPointsQueue;
        // std::queue<livox_ros_driver::CustomPoint>   _hori_points_queue;
        // std::queue<uint64>                          _hori_points_stamp_queue;
        std::vector<livox_ros_driver::CustomPoint>   _hori_points_queue;
        std::vector<uint64>                          _hori_points_stamp_queue;


        bool        _time_offset_initd    = false;
        double      _time_esti_error_th   = 400.0;
        uint64      _time_offset          = 0; //secs (velo_stamp - hori_stamp)


        // Parameter
        bool    en_timeoffset_esti         = true;   // enable time_offset estimation
        bool    en_extrinsic_esti          = true;   // enable extrinsic estimation
        bool    en_timestamp_align         = true;
        bool    _use_given_extrinsic_lidars  = false;  // using given extrinsic parameter
        bool    _use_given_timeoffset       = true;   // using given timeoffset estimation
        float   _time_start_yaw_velocity   = 0.5;    // the angular speed when time offset estimation triggered
        int     _offset_search_resolution    = 30;     // points
        int     _offset_search_sliced_points = 12000;     // points
        float   given_timeoffset          = 0;      // the given time-offset value

        // Distortion
        double _last_imu_time = 0.0;
        Eigen::Quaterniond _delta_q;

        uint64 _last_search_stamp = 0;

        double time_cost = 0;
        int processed_cnt = 0;

    public:
        LidarsParamEstimator()
        {


            // Get parameters
            ros::NodeHandle private_nh_("~");
            if (!private_nh_.getParam("enable_extrinsic_estimation",  en_extrinsic_esti))       en_extrinsic_esti = true;
            if (!private_nh_.getParam("enable_timeoffset_estimation", en_timeoffset_esti))      en_timeoffset_esti = false;
            if (!private_nh_.getParam("extri_esti_hori_integ_frames", _hori_itegrate_frames))    _hori_itegrate_frames = 10;
            if (!private_nh_.getParam("give_extrinsic_Velo_to_Hori",  _use_given_extrinsic_lidars))   _use_given_extrinsic_lidars = false;
            if (!private_nh_.getParam("time_esti_error_threshold",    _time_esti_error_th))     _time_esti_error_th = 35000.0;
            if (!private_nh_.getParam("time_esti_start_yaw_velocity", _time_start_yaw_velocity))     _time_start_yaw_velocity = 0.6;
            if (!private_nh_.getParam("give_timeoffset_Velo_to_Hori", _use_given_timeoffset))        _use_given_timeoffset = false;
            if (!private_nh_.getParam("timeoffset_Velo_to_Hori",      given_timeoffset))            given_timeoffset = 0.070;
            if (!private_nh_.getParam("timeoffset_search_resolution", _offset_search_resolution))    _offset_search_resolution = 10;
            if (!private_nh_.getParam("timeoffset_search_sliced_points", _offset_search_sliced_points)) _offset_search_sliced_points = 24000;

            if (!private_nh_.getParam("cut_raw_Hori_message_pieces", _cut_raw_message_pieces)) _cut_raw_message_pieces = 1;

            ROS_INFO_STREAM( "enable_timeoffset_estimation    : " << en_timeoffset_esti );
            ROS_INFO_STREAM( "enable_extrinsic_estimation     : " << en_extrinsic_esti );
            ROS_INFO_STREAM( "extri_esti_hori_integ_frames    : " << _hori_itegrate_frames );
            ROS_INFO_STREAM( "time_esti_error_threshold       : " << _time_esti_error_th );
            ROS_INFO_STREAM( "give_extrinsic_Velo_to_Hori     : " << _use_given_extrinsic_lidars );
            ROS_INFO_STREAM( "time_esti_start_yaw_velocity    : " <<  _time_start_yaw_velocity );
            ROS_INFO_STREAM( "give_timeoffset_Velo_to_Hori    : " <<  _use_given_timeoffset );
            ROS_INFO_STREAM( "timeoffset_Velo_to_Hori         : " <<  given_timeoffset );
            ROS_INFO_STREAM( "timeoffset_search_resolution    : " <<  _offset_search_resolution );
            ROS_INFO_STREAM( "timeoffset_search_sliced_points : " <<  _offset_search_sliced_points );
            ROS_INFO_STREAM( "cut_raw_Hori_message_pieces     : " <<  _cut_raw_message_pieces );

            sub_ouster = nh.subscribe<sensor_msgs::PointCloud2>("/ouster/points", 1000, &LidarsParamEstimator::ouster_cloud_handler, this);
            //mid1是前面的激光雷达，mid2是后面的激光雷达，mid3是中间的激光雷达
            sub_mid1 = nh.subscribe<livox_ros_driver::CustomMsg>("/livox/lidar_192_168_1_102", 100, &LidarsParamEstimator::hori_cloud_handler, this);//前
            sub_mid2 = nh.subscribe<livox_ros_driver::CustomMsg>("/livox/lidar_192_168_1_122", 100, &LidarsParamEstimator::hori_cloud_handler2, this);//后
            sub_mid3 = nh.subscribe<livox_ros_driver::CustomMsg>("/livox/lidar_192_168_1_146", 100, &LidarsParamEstimator::hori_cloud_handler3, this);//中

            pub_hori        = nh.advertise<sensor_msgs::PointCloud2>("/a_horizon", 1);
            pub_hori_livoxmsg = nh.advertise<livox_ros_driver::CustomMsg>("/a_horizon_livoxmsg", 1);
            pub_velo        = nh.advertise<sensor_msgs::PointCloud2>("/a_velo", 1);
            pub_time_hori   = nh.advertise<sensor_msgs::PointCloud2>("/a_time_hori", 1);
            pub_time_velo   = nh.advertise<sensor_msgs::PointCloud2>("/a_time_velo", 1);
            pub_velo_inFOV  = nh.advertise<sensor_msgs::PointCloud2>("/velo_fov_cloud", 1);
            pub_merged_cloud = nh.advertise<sensor_msgs::PointCloud2>("/merged_cloud", 1);
            pub_time_hori_livoxmsg = nh.advertise<livox_ros_driver::CustomMsg>("/a_time_hori_livoxmsg", 1);

            std::vector<double> vecVeloHoriExtri;
            if ( _use_given_extrinsic_lidars && private_nh_.getParam("Extrinsic_Velohori", vecVeloHoriExtri )){
                _velo_hori_tf_matrix <<    vecVeloHoriExtri[0], vecVeloHoriExtri[1], vecVeloHoriExtri[2], vecVeloHoriExtri[3],
                                    vecVeloHoriExtri[4], vecVeloHoriExtri[5], vecVeloHoriExtri[6], vecVeloHoriExtri[7],
                                    vecVeloHoriExtri[8], vecVeloHoriExtri[9], vecVeloHoriExtri[10], vecVeloHoriExtri[11],
                                    vecVeloHoriExtri[12], vecVeloHoriExtri[13], vecVeloHoriExtri[14], vecVeloHoriExtri[15];
                _hori_tf_initd = true;
                ROS_INFO_STREAM("Reveived transformation_matrix Velo-> Hori: \n" << _velo_hori_tf_matrix );
            }

            std::vector<double> vecL12OExtri;
            std::vector<double> vecL22OExtri;
            std::vector<double> vecL32OExtri;
            if(!ros::param::get("mm_PoseEstimation/Extrinsic_TM12O",vecL12OExtri )){
                vecL12OExtri = {0.460857, 0.455947, 0.760784, 0.223312,
                                -0.703766, 0.71022, 0.000669762, 0.00225066,
                                -0.540402, -0.536106, 0.648971, 0.120166,
                                0, 0, 0, 1}; // 前mid360对ouster

                ROS_WARN_STREAM("Extrinsic_Tlb unavailable ! Using default param");
            }
            if(!ros::param::get("mm_PoseEstimation/Extrinsic_TM22O",vecL22OExtri )){

                vecL22OExtri = {0.351089, 0.344676, -0.870606, -0.35431,
                                -0.710375, 0.703766, -0.00784904, -0.00602132,
                                0.610049, 0.621264, 0.491873, 0.269782,
                                0, 0, 0, 1}; // 后mid360对ouster

                ROS_WARN_STREAM("Extrinsic_Tlb unavailable ! Using default param");
            }     
            // if(!ros::param::get("mm_PoseEstimation/Extrinsic_TL22O",vecL22OExtri )){

            //     vecL22OExtri = {   -0.49789,  0.00156287,   -0.867213,   -0.392622,
            //     -0.00892625,   -0.999954,  0.00332278, -0.00168803,
            //     -0.867169,  0.00939575,    0.497882,   0.185,
            //     0,           0,           0,           1}; // 后mid360对ouster

            //     ROS_WARN_STREAM("Extrinsic_Tlb unavailable ! Using default param");
            // }
            if(!ros::param::get("mm_PoseEstimation/Extrinsic_TM32O",vecL32OExtri )){

                vecL32OExtri = {-0.707, 0.707,  0.0,   -0.0,
                                -0.707, -0.707,  0.0,   0.0,
                                0.0, 0.0,  1.0,   0.347,
                                0.0, 0.0,  0.0,   1.0}; // 后mid360对ouster

                ROS_WARN_STREAM("Extrinsic_Tlb unavailable ! Using default param");
            }
            Eigen::Matrix3f rotate_axia;
            Eigen::Matrix3f rotate_init = Eigen::Matrix3f::Identity();
            _mid1_ouster_tf_init = Eigen::Matrix4f::Identity();
            _mid2_ouster_tf_init = Eigen::Matrix4f::Identity();
            _mid3_ouster_tf_init = Eigen::Matrix4f::Identity();

            // mid360-1
            rotate_axia = Eigen::AngleAxisf(PI * 4 / 9, Eigen::Vector3f::UnitY()).toRotationMatrix();
            rotate_init = rotate_axia;
            rotate_axia = Eigen::AngleAxisf(-PI / 4, Eigen::Vector3f::UnitZ()).toRotationMatrix();
            rotate_init = rotate_init * rotate_axia;
            _mid1_ouster_tf_init.block(0,0,3,3) = rotate_init;
            _mid1_ouster_tf_init.block(0,3,3,1) << 0.223312, 0.00225066, 0.120166;
            // mid360-2
            rotate_axia = Eigen::AngleAxisf(-PI / 3, Eigen::Vector3f::UnitY()).toRotationMatrix();
            rotate_init = rotate_axia;
            rotate_axia = Eigen::AngleAxisf(3 * PI / 4, Eigen::Vector3f::UnitZ()).toRotationMatrix();
            rotate_init = rotate_init * rotate_axia;
            _mid2_ouster_tf_init.block(0,0,3,3) = rotate_init;
            _mid2_ouster_tf_init.block(0,3,3,1) << -0.35431, -0.00602132, 0.269782;
            // mid360-3
            rotate_axia = Eigen::AngleAxisf(0, Eigen::Vector3f::UnitY()).toRotationMatrix();
            rotate_init = rotate_axia;
            rotate_axia = Eigen::AngleAxisf(PI / 4, Eigen::Vector3f::UnitZ()).toRotationMatrix();
            rotate_init = rotate_init * rotate_axia;
            _mid3_ouster_tf_init.block(0,0,3,3) = rotate_init;
            _mid3_ouster_tf_init.block(0,3,3,1) << 0.0, 0.0, 0.500;

            // _mid1_ouster_tf_init <<    vecL12OExtri[0], vecL12OExtri[1], vecL12OExtri[2], vecL12OExtri[3],
            //         vecL12OExtri[4], vecL12OExtri[5], vecL12OExtri[6], vecL12OExtri[7],
            //         vecL12OExtri[8], vecL12OExtri[9], vecL12OExtri[10], vecL12OExtri[11],
            //         vecL12OExtri[12], vecL12OExtri[13], vecL12OExtri[14], vecL12OExtri[15];
            // _mid2_ouster_tf_init <<    vecL22OExtri[0], vecL22OExtri[1], vecL22OExtri[2], vecL22OExtri[3],
            //         vecL22OExtri[4], vecL22OExtri[5], vecL22OExtri[6], vecL22OExtri[7],
            //         vecL22OExtri[8], vecL22OExtri[9], vecL22OExtri[10], vecL22OExtri[11],
            //         vecL22OExtri[12], vecL22OExtri[13], vecL22OExtri[14], vecL22OExtri[15];
            // _mid3_ouster_tf_init <<    vecL32OExtri[0], vecL32OExtri[1], vecL32OExtri[2], vecL32OExtri[3],
            //         vecL32OExtri[4], vecL32OExtri[5], vecL32OExtri[6], vecL32OExtri[7],
            //         vecL32OExtri[8], vecL32OExtri[9], vecL32OExtri[10], vecL32OExtri[11],
            //         vecL32OExtri[12], vecL32OExtri[13], vecL32OExtri[14], vecL32OExtri[15];

            ROS_WARN_STREAM("Reveived transformation_matrix mid1 -> base: \n" << _mid1_ouster_tf_init );
            ROS_WARN_STREAM("Reveived transformation_matrix mid2 -> base: \n" << _mid2_ouster_tf_init );
            ROS_WARN_STREAM("Reveived transformation_matrix mid3 -> base: \n" << _mid3_ouster_tf_init );
            _mid2_mid1_tf_init = _mid1_ouster_tf_init.inverse() * _mid2_ouster_tf_init;
            ROS_WARN_STREAM("Reveived transformation_matrix mid2 -> mid1: \n" << _mid2_mid1_tf_init );
            if(_use_given_timeoffset)
            {
                ROS_INFO_STREAM("Given time offset " << given_timeoffset);
                ros::Time tmp_stamp;
                _time_offset = tmp_stamp.fromSec(given_timeoffset).toNSec();
                _time_offset_initd = true;
            }

        };

        ~LidarsParamEstimator(){};

        /**
         * @brief subscribe raw pointcloud message from Livox lidar and process the data.
         * - save the first timestamp of first message to init the timestamp
         * - Undistort pointcloud based on rotation from IMU
         * - If TF is not initlized,  Push the current undistorted message and yaw to queue;
         * - If TF is not intilized,  align two pointclouds with ICP after integrating enough frames
         * - If TF has been initized, publish aligned cloud in Horizon frame-id
         */
        void hori_cloud_handler(const livox_ros_driver::CustomMsgConstPtr& livox_msg_in)
        {
            auto tick = std::chrono::high_resolution_clock::now();
            if(!_first_velo_reveived) return; // to make sure we have velo cloud to match

            if(_first_hori){
                _hori_start_stamp = livox_msg_in->timebase; // global hori message time_base
                ROS_INFO_STREAM("Update _hori_start_stamp :" << _hori_start_stamp);
                _first_hori = false;
            }

            livox_ros_driver::CustomMsg livox_msg_in_distort(*livox_msg_in);
            // RemoveLidarDistortion( livox_msg_in, livox_msg_in_distort);

            //#### push to queue to aligh timestamp
            std::unique_lock<std::mutex> lock_hori(_mutexHoriQueue);
            _hori_queue.push(livox_msg_in_distort);
            _hori_msg_yaw_vec.push_back(_yaw_velocity);
            // if( _hori_queue.size() > 5 ) _hori_queue.pop();
            lock_hori.unlock();
            // if(_yaw_velocity > 0.6)
            //     std::cout << "Current yaw:  "<< _yaw_velocity << std::endl;

            // ###$ integrate more msgs to get extrinsic transform
            pcl::PointCloud<pcl::PointXYZI>::Ptr pointCloudIn(new  pcl::PointCloud<pcl::PointXYZI>);
            livoxToPCLCloud(livox_msg_in_distort, *pointCloudIn, _cut_raw_message_pieces);
            if(_hori_itegrate_frames > 0 && !_hori_tf_initd )
            {
                _hori_igcloud += *pointCloudIn;
                _hori_itegrate_frames--;
                ROS_INFO_STREAM("hori cloud integrating: " << _hori_itegrate_frames);
                return;
            }
            else
            {
                // Calibrate the Lidar first
                if(!_hori_tf_initd && en_extrinsic_esti){
                    Eigen::AngleAxisf init_rot_x( 0.0 , Eigen::Vector3f::UnitX());
                    Eigen::AngleAxisf init_rot_y( 0.0 , Eigen::Vector3f::UnitY());
                    Eigen::AngleAxisf init_rot_z( 0.0 , Eigen::Vector3f::UnitZ());

                    Eigen::Translation3f init_trans(0.0,0.0,0.0);
                    Eigen::Matrix4f init_tf = (init_trans * init_rot_z * init_rot_y * init_rot_x).matrix();
                    Eigen::Matrix4f mid12ouster_tf_matrix, mid22ouster_tf_matrix, mid22mid1_tg_matrix;
                    // pcl::transformPointCloud (full_cloud , cloud_out, transformation_matrix);
                    ROS_INFO("\n\n\n  Calibrate Horizon ...");

                    pcl::PointCloud<PointType> hori_temp1, hori_temp2, hori_temp3;


                    pcl::transformPointCloud (_hori_igcloud, hori_temp1, _mid1_ouster_tf_init);
                    pcl::transformPointCloud (_hori_igcloud2, hori_temp2, _mid2_ouster_tf_init);
                    pcl::transformPointCloud (_hori_igcloud3, hori_temp3, _mid3_ouster_tf_init);

                    pcl::copyPointCloud(hori_temp3, _ouster_new_cloud);

                    calibratePCLICP(hori_temp1.makeShared(), _ouster_new_cloud.makeShared(), mid12ouster_tf_matrix, true, "mid12ouster");
                    calibratePCLICP(hori_temp2.makeShared(), _ouster_new_cloud.makeShared(), mid22ouster_tf_matrix, true, "mid22ouster");
                    calibratePCLICP(hori_temp2.makeShared(), hori_temp1.makeShared(), mid22mid1_tg_matrix, true, "mid22mid1");


                    // Eigen::Matrix3f rot_matrix = hori_tf_matrix.block(0,0,3,3);
                    // Eigen::Vector3f trans_vector = hori_tf_matrix.block(0,3,3,1);

                    mid12ouster_tf_matrix = mid12ouster_tf_matrix * _mid1_ouster_tf_init;
                    mid22ouster_tf_matrix = mid22ouster_tf_matrix * _mid2_ouster_tf_init;
                    mid22mid1_tg_matrix = _mid1_ouster_tf_init.inverse() * mid22mid1_tg_matrix * _mid2_ouster_tf_init;

                    pcl::transformPointCloud (_hori_igcloud, hori_temp1, mid12ouster_tf_matrix);
                    pcl::transformPointCloud (_hori_igcloud2, hori_temp2, mid22ouster_tf_matrix);
                    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudALL (new pcl::PointCloud<pcl::PointXYZRGB>(_ouster_new_cloud.size() + hori_temp1.size() + hori_temp2.size(),1));

                    // Fill in the CloudIn data
                    for (int i = 0; i < _ouster_new_cloud.size(); i++)
                    {
                        pcl::PointXYZRGB pointin;
                        pointin.x = (_ouster_new_cloud)[i].x;
                        pointin.y = (_ouster_new_cloud)[i].y;
                        pointin.z = (_ouster_new_cloud)[i].z;
                        pointin.r = 255;
                        pointin.g = 0;
                        pointin.b = 0;
                        (*cloudALL)[i] = pointin;
                    }
                    for (int i = 0; i < hori_temp1.size(); i++)
                    {
                        pcl::PointXYZRGB pointout;
                        pointout.x = (hori_temp1)[i].x;
                        pointout.y = (hori_temp1)[i].y;
                        pointout.z = (hori_temp1)[i].z;
                        pointout.r = 0;
                        pointout.g = 255;
                        pointout.b = 0;
                        (*cloudALL)[i+_ouster_new_cloud.size()] = pointout;
                    }
                    for (int i = 0; i < hori_temp2.size(); i++)
                    {
                        pcl::PointXYZRGB pointout;
                        pointout.x = (hori_temp2)[i].x;
                        pointout.y = (hori_temp2)[i].y;
                        pointout.z = (hori_temp2)[i].z;
                        pointout.r = 0;
                        pointout.g = 0;
                        pointout.b = 255;
                        (*cloudALL)[i+_ouster_new_cloud.size() + hori_temp1.size()] = pointout;
                    }
                    pcl::io::savePCDFile<pcl::PointXYZRGB> ("/home/jcwang/dataset/icp_ICP_all.pcd", *cloudALL);
                    ROS_WARN_STREAM("transformation_matrix Mid1-> Mid3: \n"<< _mid3_ouster_tf_init.inverse() * mid12ouster_tf_matrix);
                    ROS_WARN_STREAM("transformation_matrix Mid2-> Mid3: \n"<< _mid3_ouster_tf_init.inverse() * mid22ouster_tf_matrix);
                    ROS_WARN_STREAM("transformation_matrix Mid1-> base: \n"<< mid12ouster_tf_matrix);
                    ROS_WARN_STREAM("transformation_matrix Mid2-> base: \n"<< mid22ouster_tf_matrix);
                    ROS_WARN_STREAM("transformation_matrix Mid2-> Mid1: \n"<< mid22mid1_tg_matrix);
                    std::cout << "transformation_matrix Hori-> Velo: \n"<<mid12ouster_tf_matrix << std::endl;
                    Eigen::Matrix3f rot_matrix = mid12ouster_tf_matrix.block(0,0,3,3);
                    Eigen::Vector3f trans_vector = mid12ouster_tf_matrix.block(0,3,3,1);
                    _velo_hori_tf_matrix.block(0,0,3,3) = rot_matrix.transpose();
                    _velo_hori_tf_matrix.block(0,3,3,1) =  mid12ouster_tf_matrix.block(0,3,3,1) * -1;
                    _velo_hori_tf_matrix.block(3,0,1,4) = mid12ouster_tf_matrix.block(3,0,1,4);
                    std::cout << "transformation_matrix Velo-> Hori: \n"<<_velo_hori_tf_matrix << std::endl;

                    // std::cout << "hori -> base_link " << trans_vector.transpose()
                    //     << " " << rot_matrix.eulerAngles(2,1,0).transpose() << " /" << "hori_frame"
                    //     << " /" << "livox_frame" << " 10" << std::endl;

                    // publish result
                    pcl::PointCloud<PointType>  out_cloud;
                    out_cloud += _hori_igcloud;

                    sensor_msgs::PointCloud2 hori_msg;
                    pcl::toROSMsg(out_cloud, hori_msg);
                    hori_msg.header.stamp = ros::Time::now();
                    hori_msg.header.frame_id = "lio_world";
                    pub_hori.publish(hori_msg);

                    _hori_tf_initd = true;
                }
            }
        }
    /**
     * @brief subscribe raw pointcloud message from Livox lidar and process the data.
     * - save the first timestamp of first message to init the timestamp
     * - Undistort pointcloud based on rotation from IMU
     * - If TF is not initlized,  Push the current undistorted message and yaw to queue;
     * - If TF is not intilized,  align two pointclouds with ICP after integrating enough frames
     * - If TF has been initized, publish aligned cloud in Horizon frame-id
     */
    void hori_cloud_handler3(const livox_ros_driver::CustomMsgConstPtr& livox_msg_in)
    {
        auto tick = std::chrono::high_resolution_clock::now();
        if(!_first_velo_reveived) _first_velo_reveived = true; // to make sure we have velo cloud to match

        if(_first_hori3){
            _hori_start_stamp3 = livox_msg_in->timebase; // global hori message time_base
            ROS_INFO_STREAM("Update _hori_start_stamp3 :" << _hori_start_stamp3);
            _first_hori3 = false;
        }

        livox_ros_driver::CustomMsg livox_msg_in_distort(*livox_msg_in);
        // RemoveLidarDistortion( livox_msg_in, livox_msg_in_distort);
        // ###$ integrate more msgs to get extrinsic transform
        pcl::PointCloud<pcl::PointXYZI>::Ptr pointCloudIn(new  pcl::PointCloud<pcl::PointXYZI>);
        livoxToPCLCloud(livox_msg_in_distort, *pointCloudIn, _cut_raw_message_pieces);
        if(_hori_itegrate_frames3 > 0 && !_hori_tf_initd )
        {
            _hori_igcloud3 += *pointCloudIn;
            _hori_itegrate_frames3--;
            ROS_INFO_STREAM("hori cloud integrating: " << _hori_itegrate_frames3);
            return;
        }
    }

        /**
         * @brief subscribe raw pointcloud message from Livox lidar and process the data.
         * - save the first timestamp of first message to init the timestamp
         * - Undistort pointcloud based on rotation from IMU
         * - If TF is not initlized,  Push the current undistorted message and yaw to queue;
         * - If TF is not intilized,  align two pointclouds with ICP after integrating enough frames
         * - If TF has been initized, publish aligned cloud in Horizon frame-id
         */
        void hori_cloud_handler2(const livox_ros_driver::CustomMsgConstPtr& livox_msg_in)
        {
            auto tick = std::chrono::high_resolution_clock::now();
            if(!_first_velo_reveived) return; // to make sure we have velo cloud to match

            if(_first_hori2){
                _hori_start_stamp2 = livox_msg_in->timebase; // global hori message time_base
                ROS_INFO_STREAM("Update _hori_start_stamp2 :" << _hori_start_stamp2);
                _first_hori2 = false;
            }

            livox_ros_driver::CustomMsg livox_msg_in_distort(*livox_msg_in);
            // RemoveLidarDistortion( livox_msg_in, livox_msg_in_distort);
            // ###$ integrate more msgs to get extrinsic transform
            pcl::PointCloud<pcl::PointXYZI>::Ptr pointCloudIn(new  pcl::PointCloud<pcl::PointXYZI>);
            livoxToPCLCloud(livox_msg_in_distort, *pointCloudIn, _cut_raw_message_pieces);
            if(_hori_itegrate_frames2 > 0 && !_hori_tf_initd )
            {
                _hori_igcloud2 += *pointCloudIn;
                _hori_itegrate_frames2--;
                ROS_INFO_STREAM("hori cloud integrating: " << _hori_itegrate_frames2);
                return;
            }
        }

        /**
         * @brief Subscribe pointcloud from Velodyne
         * - save the first timestamp of first message to init the timestamp
         * - undistort pointcloud based on rotation from IMU
         * - select Velo points in same FOV with Hori
         * -
         * @param pointCloudIn
         */
        void ouster_cloud_handler(const sensor_msgs::PointCloud2ConstPtr& pointCloudIn)
        {

            if(!_first_velo_reveived) _first_velo_reveived = true;

            pcl::PointCloud<PointType>  full_cloud_in;
            pcl::fromROSMsg(*pointCloudIn, full_cloud_in);


            // ################ select Velo points in same FOV with Hori #################
            int cloudSize = full_cloud_in.points.size();
            pcl::PointCloud<pcl::PointXYZI> ouster_fovs_cloud;
            float startOri = -atan2(full_cloud_in.points[0].y, full_cloud_in.points[0].x);
            float endOri = -atan2(full_cloud_in.points[cloudSize - 1].y,
                    full_cloud_in.points[cloudSize - 1].x) +
                    2 * M_PI;

            if (endOri - startOri > 3 * M_PI)
                endOri -= 2 * M_PI;
            else if (endOri - startOri < M_PI)
                endOri += 2 * M_PI;

            // ROS_INFO_STREAM("Velodyne Lidar start angle: " <<  startOri << " | end angle : " << endOri << " | range: " << endOri - startOri);

            pcl::PointCloud<PointType> undistort_cloud;
            pcl::PointXYZI  point;
            bool halfPassed = false;
            for (int i = 0; i < cloudSize; i++)
            {
                point.x = full_cloud_in.points[i].x;
                point.y = full_cloud_in.points[i].y;
                point.z = full_cloud_in.points[i].z;

                float ori = -atan2(point.y, point.x);
                if (!halfPassed)
                {
                    if (ori < startOri - M_PI / 2)
                        ori += 2 * M_PI;
                    else if (ori > startOri + M_PI * 3 / 2)
                        ori -= 2 * M_PI;

                    if (ori - startOri > M_PI)
                        halfPassed = true;
                }
                else {
                    ori += 2 * M_PI;
                    if (ori < endOri - M_PI * 3 / 2)
                        ori += 2 * M_PI;
                    else if (ori > endOri + M_PI / 2)
                        ori -= 2 * M_PI;
                }

                float relTime = (ori - startOri) / (endOri - startOri);
                point.intensity = relTime;

                // 不需要判断是否在水平视场内
                 if( ( ori > -2.356 && ori < 2.356 ) ||
                     ori > -2.356 + 2*M_PI && ori < 2.356 +2*M_PI)
                {
                    // velo_fovs_cloud.push_back(point);
                    undistort_cloud.push_back(point);
                    // if(undistort_cloud.size() == 1)
                    //     ROS_INFO_STREAM("First Points in FOV, start angle : " << ori << "  | relTime" << relTime);
                }
            }
        }

};

int
main(int argc, char **argv)
{
    ros::init(argc, argv, "Lidar_Calibrate");

    LidarsParamEstimator swo;
    ROS_INFO("\033[1;32m---->\033[0m Lidar Calibrate Started.");

    ros::Rate r(10);
    while(ros::ok()){

        ros::spinOnce();
        r.sleep();

    }


    return 0;
}
