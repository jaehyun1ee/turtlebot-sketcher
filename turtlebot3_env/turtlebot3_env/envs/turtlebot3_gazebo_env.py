import sys
import os
import subprocess
from math import radians, copysign, sqrt, pow, pi, atan2
import numpy as np

import gym
from gym import spaces

import rospy
import tf
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from geometry_msgs.msg import Twist, Point, Quaternion
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import GetModelState, SetModelState
   
class Turtlebot3GazeboEnv(gym.Env):
    def __init__(self):
        # initialize node
        rospy.init_node('turtlebot3_env', anonymous=False)

        # launch gazebo
        ros_path = os.path.dirname(subprocess.check_output(["which", "roscore"]))
        #launchfile = "/home/dongjoo/catkin_ws/src/cs470-stroker/turtlebot3_simulations/turtlebot3_gazebo/launch/turtlebot3_empty_world.launch"
        launchfile = str(os.getcwd()) + "/turtlebot3_simulations/turtlebot3_gazebo/launch/turtlebot3_empty_world.launch"         
        os.environ["TURTLEBOT3_MODEL"] = "burger"
        subprocess.Popen([sys.executable, os.path.join(ros_path, b"roslaunch"), "-p", "11311", launchfile])
        print ("Gazebo launched!")

        # initialize agent
        self.agent = Agent()

        # initialize environment (MDP definition)
        self.observation_space = spaces.Dict({
            "agent" : spaces.Box(low=np.array([-np.inf, -np.inf, -np.pi]), high=np.array([np.inf, np.inf, np.pi]), dtype=np.float64), # x-coord, y-coord, and rotation angle in radians
            "target" : spaces.Box(low=np.array([-np.inf, -np.inf]), high=np.array([np.inf, np.inf]), dtype=np.float32),        
        })
        self.action_space = spaces.Box(low=np.array([-1.5, -1.5]), high=np.array([1.5, 1.5]), dtype=np.float64) # linear-x, and angular-z
        self.state = {
            "agent" : np.array([0, 0, 0]),
            "target" : np.random.rand(2) * 5
        } 

    def step(self, action):
        cmd = Twist()
        cmd.linear.x = action[0]
        cmd.angular.z = action[1]
        self.state["agent"] = self.agent.move(cmd)

        return self.state, 0, 0, {}

    def stop(self):
        self.state["agent"] = self.agent.move(Twist())

    def reset(self):
        self.state["agent"] = self.agent.reset()
        self.state["target"] = np.random.rand(2) * 5

        return self.state, {}

    def close(self):
         # find gzclient and gzserver 
        tmp = os.popen("ps -Af").read()
        gzclient_count = tmp.count('gzclient')
        gzserver_count = tmp.count('gzserver')

        # kill gzclient and gzserver
        if gzclient_count > 0:
            os.system("killall -9 gzclient")
        if gzserver_count > 0:
            os.system("killall -9 gzserver")
        if (gzclient_count or gzserver_count > 0):
            os.wait()

class Agent():
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

        # initialize
        self.rate = rospy.Rate(10)
        self.model_name = "turtlebot3_burger"

    def move(self, cmd):
        self.cmd_vel.publish(cmd)
        self.rate.sleep()

        return self.get_state()

    def reset(self):
        cmd = ModelState()
        cmd.model_name = self.model_name
        orientation = quaternion_from_euler(0, 0, np.random.rand() * np.pi * 2 - np.pi)
        cmd.pose.orientation.x = orientation[0]
        cmd.pose.orientation.y = orientation[1]
        cmd.pose.orientation.z = orientation[2]
        cmd.pose.orientation.w = orientation[3]
        self.set_model_state(cmd)

        return self.get_state()

    def get_state(self):
        resp = self.get_model_state(self.model_name, "")
        x = resp.pose.position.x
        y = resp.pose.position.y
        rot = euler_from_quaternion([resp.pose.orientation.x, resp.pose.orientation.y, resp.pose.orientation.z, resp.pose.orientation.w])

        return np.array([ x, y, rot[2] ])
