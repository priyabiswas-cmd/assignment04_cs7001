#!/usr/bin/env python3
"""
Bring up the parallel-parking world with a TurtleBot3 Burger.

    ros2 launch parking_bc parking_demo.launch.py
    ros2 launch parking_bc parking_demo.launch.py world:=parking_narrow.world
    ros2 launch parking_bc parking_demo.launch.py x:=0.9 y:=0.75 yaw:=3.14159
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('parking_bc')
    gazebo_ros = get_package_share_directory('gazebo_ros')
    tb3_desc = get_package_share_directory('turtlebot3_description')
    tb3_gazebo = get_package_share_directory('turtlebot3_gazebo')

    world = LaunchConfiguration('world')
    x = LaunchConfiguration('x')
    y = LaunchConfiguration('y')
    yaw = LaunchConfiguration('yaw')
    use_sim_time = LaunchConfiguration('use_sim_time')

    urdf = os.path.join(tb3_desc, 'urdf', 'turtlebot3_burger.urdf')
    with open(urdf, 'r') as f:
        robot_description = f.read()

    return LaunchDescription([
        DeclareLaunchArgument('world', default_value='parking_grid.world'),
        # Default start pose: alongside the curb, offset out from the bay.
        DeclareLaunchArgument('x', default_value='0.75'),
        DeclareLaunchArgument('y', default_value='0.55'),
        DeclareLaunchArgument('yaw', default_value='3.14159'),
        DeclareLaunchArgument('use_sim_time', default_value='true'),

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
