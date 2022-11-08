import sys
import os
#import gym
import subprocess
import rospy
from geometry_msgs.msg import Twist, Point, Quaternion
import tf
from math import radians, copysign, sqrt, pow, pi, atan2
from tf.transformations import euler_from_quaternion
import numpy as np
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import GetModelState, SetModelState

   
class SimpleEnv():
    def __init__(self):
        rospy.init_node('turtlebot3_pointop_key', anonymous=False)
        self.lower = Lower()
        ros_path = os.path.dirname(subprocess.check_output(["which", "roscore"]))

        launchfile = "/home/dongjoo/catkin_ws/src/cs470-stroker/turtlebot3_simulations/turtlebot3_gazebo/launch/turtlebot3_empty_world.launch"
        
        os.environ["TURTLEBOT3_MODEL"] = "burger"
        subprocess.Popen([sys.executable, os.path.join(ros_path, b"roslaunch"), "-p", "11311", launchfile])
        print ("Gazebo launched!")
        self.state = [0, 0, 0]

    def observation(self):
        return self.lower.get_state()

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

        lower.command(cmd)

        self.state = self.observation()

        return self.state, 0, 0, {}

    def reset(self):
        state_msg = ModelState()
        state_msg.model_name = "turtlebot3_burger"
        state_msg.pose.position.x = 0
        state_msg.pose.position.y = 0
        state_msg.pose.position.z = 0
        state_msg.pose.orientation.x = 0
        state_msg.pose.orientation.y = 0
        state_msg.pose.orientation.z = 0
        state_msg.pose.orientation.w = 0

        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            resp = set_state(state_msg)

        except rospy.ServiceException as err:
            print(err)

        state = self.observation()

        return state

class Lower():
    def __init__(self):
        self.cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        self.tf_listener = tf.TransformListener()
        self.odom_frame = 'odom'
        self.model = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.r = rospy.Rate(10)

    def command(self, cmd):
        self.cmd_vel.publish(cmd)
        self.r.sleep()
        return self.get_state()

    def get_state(self):
        resp = self.model("turtlebot3_burger", "")
        x = resp.pose.position.x
        y = resp.pose.position.y
        rot = euler_from_quaternion([resp.pose.orientation.x, resp.pose.orientation.y, resp.pose.orientation.z, resp.pose.orientation.w])

        return [ x, y, rot[2] ]

    def get_odom(self):
        try:
            self.tf_listener.waitForTransform(self.odom_frame, 'base_footprint', rospy.Time(), rospy.Duration(1.0))
            self.base_frame = 'base_footprint'
        except (tf.Exception, tf.ConnectivityException, tf.LookupException):
            try:
                self.tf_listener.waitForTransform(self.odom_frame, 'base_link', rospy.Time(), rospy.Duration(1.0))
                self.base_frame = 'base_link'
            except (tf.Exception, tf.ConnectivityException, tf.LookupException):
                rospy.loginfo("Cannot find transform between odom and base_link or base_footprint")
                rospy.signal_shutdown("tf Exception")

        try:
            (trans, rot) = self.tf_listener.lookupTransform(self.odom_frame, self.base_frame, rospy.Time(0))
            rotation = euler_from_quaternion(rot)

        except (tf.Exception, tf.ConnectivityException, tf.LookupException):
            rospy.loginfo("TF Exception")
            return

        return (Point(*trans), rotation[2])


if __name__ == "__main__":
    env = SimpleEnv()
    env.step(0)
