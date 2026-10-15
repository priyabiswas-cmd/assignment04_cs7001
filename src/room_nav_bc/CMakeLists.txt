cmake_minimum_required(VERSION 3.10)
project(room_nav_bc)

find_package(ament_cmake REQUIRED)

install(DIRECTORY worlds launch config
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/record_demo_room.py
  scripts/expert_demo_room.py
  scripts/verify_expert_room.py
  scripts/bc_dataset_room.py
  scripts/bc_train_room.py
  scripts/bc_policy_node_room.py
  scripts/show_starts_room.py
  scripts/analyse_rollouts_room.py
  scripts/dagger_room.py
  DESTINATION lib/${PROJECT_NAME}
)
# More executable scripts are added here as they're built (dataset
# building, training, the scripted expert, DAgger loop - see scripts/).

ament_package()
