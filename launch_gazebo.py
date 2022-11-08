i11311mport sys
import os
import subprocess

if __name__ == '__main__':
    # self.port = os.environ.get("ROS_PORT_SIM", "11311")
    ros_path = os.path.dirname(subprocess.check_output(["which", "roscore"]))

    # NOTE: It doesn't make sense to launch a roscore because it will be done when spawing Gazebo, which also need
    #   to be the first node in order to initialize the clock.
    # # start roscore with same python version as current script
    # self._roscore = subprocess.Popen([sys.executable, os.path.join(ros_path, b"roscore"), "-p", self.port])
    # time.sleep(1)
    # print ("Roscore launched!")

    subprocess.Popen([sys.executable, os.path.join(ros_path, b"roslaunch"), "-p", "11311", "/home/jaehyun/catkin_ws/src/turtlebot3_simulations/turtlebot3_gazebo/launch/turtlebot3_empty_world.launch"])
    print ("Gazebo launched!")
