import sys
import os
import subprocess
import time
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
        launchfile = str(os.getcwd()) + "/turtlebot3_simulations/turtlebot3_gazebo/launch/turtlebot3_empty_world.launch"         
        os.environ["TURTLEBOT3_MODEL"] = "burger"
        subprocess.Popen([sys.executable, os.path.join(ros_path, b"roslaunch"), "-p", "11311", launchfile])
        print ("Gazebo launched!")

        # initialize agent
        self.agent = Agent()

        # initialize environment (MDP definition)
        self.observation_space = spaces.Box(low=np.array([-1, -1, -np.pi, -1, -1, 0, -1]), high=np.array([1, 1, np.pi, 1, 1, sqrt(2), 1]), shape=(7,), dtype=np.float64) # (x_agent, y_agent, rot_agent, x_goal, y_goal, dist_to_goal, similarity_to_goal)
        self.action_space = spaces.Box(low=np.array([-np.pi, -2]), high=np.array([np.pi, 2]), shape=(2,), dtype=np.float32) # linear-x, and angular-z
        self.state = {
            "agent" : np.array([0, 0, 0]),
            "target" : self.random_vector(),
            "info" : np.array([0, 0]), # will be updated below
        } 
        self.state["info"] = self.get_info()

    """
    MDP Logic
    """

    # perform one step in an episode
    def step(self, action):
        # perform action on environment
        # update state
        self.state["agent"] = self.agent.move(action)
        self.state["info"] = self.get_info()
        self.agent.move([0, 0])  

        # compute reward
        reward = self.compute_reward()
        """
        target = self.state["target"]
        agent = self.state["agent"]
        print(f"distance : {self.dist_to_goal()}\tgoal : {target}\tagent : {agent}")
        """

        return self.state_to_obs(), reward, True, {}
    
    # reward function
    def compute_reward(self):
        return -3 * self.dist_to_goal() + abs(self.similarity_to_goal())

    # reset the environment
    def reset(self):
        self.state["agent"] = self.agent.reset()
        self.state["target"] = self.random_vector()
        self.state["info"] = self.get_info()

        return self.state_to_obs()

    # stop the agent
    def stop(self):
        self.state["info"] = self.get_info()

    """
    HELPER FUNCTIONS
    """

    # state (dict) to observation (np array)
    def state_to_obs(self):
        return np.concatenate([self.state["agent"], self.state["target"], self.state["info"]])

    # return updated (recomputed) info
    def get_info(self):
        return [ self.dist_to_goal(), self.similarity_to_goal() ]

    # returns the distance between the current agent and the goal
    def dist_to_goal(self):
        dx = self.state["target"][0] - self.state["agent"][0]
        dy = self.state["target"][1] - self.state["agent"][1]
        return sqrt(dx ** 2 + dy ** 2)
    
    # returns the cosine similarity between the current agent and the goal
    def similarity_to_goal(self):
        va = self.state["agent"][:2]
        vt = self.state["target"]
        if np.allclose(va, np.zeros(2)):
            rot = self.state["agent"][2]
            va = np.array([np.cos(rot), np.sin(rot)])
        return np.dot(va, vt) / (np.linalg.norm(va) * np.linalg.norm(vt))

    # returns a random 2D vector in a circle with radius 0
    def random_vector(self):
        rand = np.zeros(2)
        while np.allclose(rand, np.zeros(2)) or np.linalg.norm(rand) > 1:
            rand = np.random.rand(2) * 2 - 1
        return rand
 
    # find and kill gazebo
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
        self.model_name = "turtlebot3_burger"
        self.orientations = np.array([i * np.pi / 4 for i in range(8)])

    # issue a move command to the turtlebot and returns its updated position
    def move(self, action):
        cmd_turn = Twist()
        cmd_turn.angular.z = 1 if action[0] > 0 else -1
        self.cmd_vel.publish(cmd_turn)
        rospy.sleep(abs(action[0]))
        self.cmd_vel.publish(Twist())

        cmd_move = Twist()
        cmd_move.linear.x = 0.5 if action[0] > 0 else -0.5
        self.cmd_vel.publish(cmd_move)
        rospy.sleep(abs(action[1])+0.5)
        self.cmd_vel.publish(Twist())

        return self.get_state()

    # resets the turtlebot in the origin with a random orientation
    def reset(self):
        cmd = ModelState()
        cmd.model_name = self.model_name
        orientation = quaternion_from_euler(0, 0, np.random.choice(self.orientations))
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
