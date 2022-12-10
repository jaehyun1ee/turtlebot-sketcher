# Reproducing a Line Drawing with Turtlebot3 Sketcher
#### 2022 Fall CS470 Team 2 Sujeburger

# About Our Project

We have implemented a Turtlebot3 Sketcher, that reproduces a given line drawing (or doodle) with the robot's trajectory as follows.

Here is the [poster](poster.pdf) that explains our work in detail.

# Demo Video of Our Model

[![Demo of our Sketcher](http://img.youtube.com/vi/1Hjz8KOL0RE/0.jpg)](https://youtu.be/1Hjz8KOL0RE)

# How to Reproduce our Work

This section explains how to run an example of our model.

The following steps will lead to a run of, `run_benchmark.py`, which will run the model given the four inputs stored in `benchmark`.

## Environment

### Our Environment

Ubuntu 20.04 LTS (ARM64), ROS Noetic running on MAC M1 Parallels VM

**This work has heavy dependencies, so we recommend following the requirements with a vanilla Ubuntu.**

### Docker Image

You may use the following docker image which has all the dependencies installed.

```
docker pull wenko99/cs470:sketcher
```

**However it is also based on ARM64, so it will likely not work in other architectures.**

**So again, we recommend following the requirements written below on a vanilla Ubuntu.**

## Installing ROS

Please follow the quickstart guide in [ROBOTIS EMANUAL](https://emanual.robotis.com/docs/en/platform/turtlebot3/quick-start/) for **Noetic** version, from 3.1.1 to 3.1.4.

## Cloning this Repository

Please initialize your workspace in `~/catkin_ws/src` as follows.

```bash
$ source /opt/ros/noetic/setup.sh
$ mkdir -p ~/catkin_ws/src
$ cd ~/catkin_ws/src/
$ catkin_init_workspace
$ rm -rf CMakeLists.txt
$ git clone https://github.com/wenko99/cs470-stroker.git .
$ git submodule update --init
```

And build the package,

```bash
# in ~/catkin_ws/src,
$ cd ..
$ source /opt/ros/noetic/setup.bash
$ catkin_make
```

## Installing Python3 Dependencies

First install `pip3`,

```bash
$ sudo apt install python3-pip
```

Then install python3 packages,

```bash
$ pip install gym
$ pip install stable_baselines3
$ pip install rdp
$ pip install bresenham
$ pip install tqdm
$ pip install tensorflow
$ pip install -e turtlebot3_env 
```

## Unzip the Model

Unzip the `model.zip` file that contains our model description,

```bash
# in ~/catkin_ws/src,
$ unzip model.zip
```

## Build the Package Again

```bash
# in ~/catkin_ws/src,
$ cd ..
$ catkin_make
$ cd src
```

## Run `roscore`

In another terminal, run

```bash
# another terminal,
$ roscore
```

## Speed Up Gazbo Simulation (Recommended)

We recommend to speed up the simulation time in Gazebo for faster run of our model.

In `turtlebot3_simulation/turtlebot3_gazebo/worlds/empty.world`,

Change the `max_step_size` from `0.001` to `0.005`. (It accelerates the simulation times by 5 times the real time.)

```xml
<max_step_size>0.005</max_step_size>
```

## Finally, Run Benchmark

### Running `run_benchmark.py`

Run the benchmark tests for our model with,

```console
# in ~/catkin_ws/src,
$ python3 run_benchmark.py
```

### Possible Error and Fix

Executing the above command will likely produce an error (in ARM64 architecture) which should look like,

```bash
OSError: /home/{USERNAME}/.local/lib/python3.8/site-packages/torch/lib/../../torch.libs/libgomp-d22c30c5.so.1.0.0: cannot allocate memory in static TLS block
```

It is likely to be solvable by entering the following command. Related to the [Issue](https://github.com/opencv/opencv/issues/14884).

Export a variable `LD_PRELOAD` with the path emitted in the error message.

```bash
$ export LD_PRELOAD=/home/{USERNAME}/.local/lib/python3.8/site-packages/torch/lib/../../torch.libs/libgomp-d22c30c5.so.1.0.0
```

Then, a re-run of the `run_benchmark.py` should work properly.

### What the Output Should Look Like

It runs our `Commander` and `Stroker` given the input images in the `benchmark` folder.

After the run, each folders in `benchmark` should contain,

1. `raw.png` : the original input image
2. `commander.png` : the image that the `Commander` has planned, in its imaginary canvas
3. `stroker.png` : the image that the `Stroker` has produced on the Gazebo world, given the ground-truth strokes
4. `commander-and-stroker.png` : the output of our integrated model, where the `Commander` plans the strokes and the `Stroker` actuates the strokes

# Contact Information

If there is any difficulty reproducing our work, please contact via email 99jaehyunlee@kaist.ac.kr
