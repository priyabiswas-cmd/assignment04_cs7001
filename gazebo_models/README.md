# Gazebo models

Copy of `~/.gazebo/models` used with this workspace (cafe_table, first_2015_trash_can,
ground_plane, mailbox, sun, table_marble).

To use on a new machine, either copy them into Gazebo's model folder:

```bash
mkdir -p ~/.gazebo/models && cp -r gazebo_models/* ~/.gazebo/models/
```

or point Gazebo at this folder:

```bash
export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:~/colcon_ws/gazebo_models
```
