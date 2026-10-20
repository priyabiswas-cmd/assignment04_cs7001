#!/usr/bin/env python3
"""
Bring up the single-room navigation world with a camera-equipped
TurtleBot3 Burger (turtlebot3_burger_cam - has /odom, /scan AND
/camera/image_raw; the learned policy will only ever use the camera and
odometry, never the lidar or ground truth - see the project discussion).

    ros2 launch room_nav_bc room_demo.launch.py
    ros2 launch room_nav_bc room_demo.launch.py x:=0.2 y:=0.4 yaw:=90
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('room_nav_bc')
    gazebo_ros = get_package_share_directory('gazebo_ros')
    tb3_gazebo = get_package_share_directory('turtlebot3_gazebo')

    world = LaunchConfiguration('world')
    x = LaunchConfiguration('x')
    y = LaunchConfiguration('y')
    yaw = LaunchConfiguration('yaw')
    use_sim_time = LaunchConfiguration('use_sim_time')

    # turtlebot3_burger_cam's urdf lives under turtlebot3_gazebo, not
    # turtlebot3_description (that package only ships the plain Burger).
    urdf = os.path.join(tb3_gazebo, 'urdf', 'turtlebot3_burger_cam.urdf')
    with open(urdf, 'r') as f:
        robot_description = f.read()

    return LaunchDescription([
        DeclareLaunchArgument('world', default_value='room_single.world'),
        # Default start pose: see worlds/generate_room.py START_POSE.
        DeclareLaunchArgument('x', default_value='0.0'),
        DeclareLaunchArgument('y', default_value='0.4'),
        DeclareLaunchArgument('yaw', default_value='1.5707963'),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        # gzclient is only the visualisation window - sensors (including
        # the camera) render inside gzserver either way, so this is purely
        # a wall-clock speedup. Useful for policy-gradient training, which
        # runs many episodes and never looks at the camera feed anyway.
        DeclareLaunchArgument('gui', default_value='true'),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(gazebo_ros, 'launch', 'gzserver.launch.py')),
            launch_arguments={
                'world': PathJoinSubstitution([pkg, 'worlds', world]),
                'verbose': 'true',
            }.items(),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(gazebo_ros, 'launch', 'gzclient.launch.py')),
            condition=IfCondition(LaunchConfiguration('gui')),
        ),

        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'robot_description': robot_description,
            }],
        ),

        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            output='screen',
            arguments=[
                '-entity', 'burger',
                '-file', os.path.join(tb3_gazebo, 'models',
                                      'turtlebot3_burger_cam', 'model.sdf'),
                '-x', x, '-y', y, '-z', '0.01', '-Y', yaw,
            ],
        ),
    ])
