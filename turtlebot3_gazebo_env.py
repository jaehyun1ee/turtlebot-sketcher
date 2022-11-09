import sys
import os
#import gym
import subprocess
from math import radians, copysign, sqrt, pow, pi, atan2
import numpy as np

import rospy
import tf
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import Twist, Point, Quaternion
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import GetModelState, SetModelState
   
class SimpleEnv():
    def __init__(self):
        # initialize node
        rospy.init_node('turtlebot3_pointop_key', anonymous=False)

        # launch gazebo
        ros_path = os.path.dirname(subprocess.check_output(["which", "roscore"]))
        #launchfile = "/home/dongjoo/catkin_ws/src/cs470-stroker/turtlebot3_simulations/turtlebot3_gazebo/launch/turtlebot3_empty_world.launch"
        launchfile = "/home/jaehyun/catkin_ws/src/turtlebot3_simulations/turtlebot3_gazebo/launch/turtlebot3_empty_world.launch"
        os.environ["TURTLEBOT3_MODEL"] = "burger"
        subprocess.Popen([sys.executable, os.path.join(ros_path, b"roslaunch"), "-p", "11311", launchfile])
        print ("Gazebo launched!")

        # initialize
        self.lower = Lower()
        self.state = [0, 0, 0]

    def step(self, action):
        cmd = Twist()
        if action == 0: #FORWARD
            cmd.linear.x = 0.3
            cmd.angular.z = 0
        elif action == 1: #LEFT
            cmd.linear.x = 0.05
            cmd.angular.z = 0.3
        elif action == 2: #RIGHT
            cmd.linear.x = 0.05
            cmd.angular.z = -0.3

        self.state = self.lower.command(cmd)
        print("action : " + str(action))
        print("state : " + str(self.state))

        return self.state, 0, 0, {}

    def reset(self):
        return self.lower.reset()

    def shutdown(self):
         # kill gzclient and gzserver 
        tmp = os.popen("ps -Af").read()
        gzclient_count = tmp.count('gzclient')
        gzserver_count = tmp.count('gzserver')

        if gzclient_count > 0:
            os.system("killall -9 gzclient")
        if gzserver_count > 0:
            os.system("killall -9 gzserver")

        if (gzclient_count or gzserver_count > 0):
            os.wait()

class Lower():
    def __init__(self):
        # for issuing command
        self.cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        while self.cmd_vel.get_num_connections() < 1:
            pass
        # for getting turtlebot position
        rospy.wait_for_service('/gazebo/get_model_state')
        self.get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        # for resetting turtlebot position
        rospy.wait_for_service('/gazebo/set_model_state')
        self.set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        self.rate = rospy.Rate(10)

    def command(self, cmd):
        self.cmd_vel.publish(cmd)
        self.rate.sleep()

        return self.get_state()

    def reset(self):
        cmd = ModelState()
        cmd.model_name = "turtlebot3_burger"
        self.set_model_state(cmd)

        return self.get_state()

    def get_state(self):
        resp = self.get_model_state("turtlebot3_burger", "")
        x = resp.pose.position.x
        y = resp.pose.position.y
        rot = euler_from_quaternion([resp.pose.orientation.x, resp.pose.orientation.y, resp.pose.orientation.z, resp.pose.orientation.w])

        return [ x, y, rot[2] ]

if __name__ == "__main__":
    env = SimpleEnv()
    for i in range(100):
        env.step(i % 3)
    env.lower.command(Twist())
    input("ENTER for RESET") # to wait before reset 
    env.reset()
    input("ENTER for SHUTDOWN") # to wait before shutdown
    env.shutdown()
