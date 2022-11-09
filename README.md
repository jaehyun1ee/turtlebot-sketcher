# CS470 Turtlebot3 Sketcher - Stroker

Ubuntu 20.04 LTS, ROS Noetic, and Parallels for Mac M1

## Initialize submodules

After cloning this repository,

```
git submodule update --init
```

to clone the turtlebot3_simulations submodule

## To run this example,

In `~/catkin_ws`,

### Build

```
catkin_make
```

### In one terminal, run `roscore`

```
roscore
```

### In another terminal, run `launch_env.py`

```
python3 launch_env.py
```

### To turn off the gazebo gui,

In `turtlebot3_empty_world.launch`, change

```
<arg name="gui" value="true"/> to, <arg name="gui" value="false"/> 
```

```
export TURTLEBOT3_MODEL=burger
roslaunch turtlebot3_example turtlebot3_pointop_key.launch
```
