cmake_minimum_required(VERSION 3.10)
project(parking_bc)

find_package(ament_cmake REQUIRED)

install(DIRECTORY worlds launch config
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/record_demo.py
  scripts/start_poses.py
  scripts/bc_dataset.py
  scripts/bc_train.py
  scripts/bc_policy_node.py
  scripts/evaluate_parking.py
  DESTINATION lib/${PROJECT_NAME}
)

# odometry+lidar variant - separate scripts, separate data/dataset/policy,
# same world/launch/config as above
install(PROGRAMS
  scripts_lidar/record_demo_lidar.py
  scripts_lidar/lidar_gap.py
  scripts_lidar/bc_dataset_lidar.py
  scripts_lidar/bc_train_lidar.py
  scripts_lidar/bc_policy_node_lidar.py
  scripts_lidar/bc_dataset_lidar_raw.py
  scripts_lidar/bc_policy_node_lidar_raw.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
