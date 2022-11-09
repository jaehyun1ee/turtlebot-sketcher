# CS470 Turtlebot3 Sketcher - Stroker

## Initialize submodules

after cloning this repository,

```
git submodule update --init
```

to clone the turtlebot3_simulations submodule

## To run this example,

In `~/catkin_ws`,

1. Build

```
catkin_make
```

2. In one terminal, run `roscore`

```
roscore
```

3. In another terminal, run turtlebot3 gazebo simulation

```
export TURTLEBOT3_MODEL=burger
roslaunch turtlebot3_gazebo turtlebot3_empty_world.launch
```

4. In another terminal, run point operation

```
export TURTLEBOT3_MODEL=burger
roslaunch turtlebot3_example turtlebot3_pointop_key.launch
```
