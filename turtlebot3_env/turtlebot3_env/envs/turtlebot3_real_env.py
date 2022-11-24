import sys
import os
import subprocess
import time
from math import radians, copysign, sqrt, pow, pi, atan2, fabs, modf
import numpy as np
import matplotlib.pyplot as plt

import gym
from gym import spaces

import rospy
import tf
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from geometry_msgs.msg import Twist, Point, Quaternion
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import GetModelState, SetModelState
   
class Turtlebot3RealEnv(gym.Env):
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
        self.agent = Agent(5) 

        # initialize environment (MDP definition)
        self.observation_space = spaces.Box(low=np.array([-1, -1, -np.pi, 0, -1.5, -1, -1, -np.pi]), high=np.array([1, 1, np.pi, 0.15, 1.5, 1, 1, np.pi]), shape=(8,), dtype=np.float64) # (x_agent, y_agent, rot_agent, x_goal, y_goal, dist_to_goal, similarity_to_goal)
        #self.action_space = spaces.MultiDiscrete(np.array([ 5, 5 ])) # rotate and forward      
        self.action_space = spaces.Discrete(5) # rotate and forward      
        self.state = {
            "agent" : np.array([0., 0., 0.]), # agent_x, agent_y, agent_rot
            "target" : np.array([0., 0., 0.]), # target_x, target_y, target_rot
            "info" : np.array([0., 0., 0.]), # will be updated below
        } 
        self.state["info"] = self.get_info()

        # initialize variables
        self.ep_len = 0
        self.dist_init = self.dist_to_goal()
        self.dist_min = self.dist_init
        self.trajectory = [ self.state["agent"][:2] ]
        self.targets = [ self.state["target"][:2] ]

    def init_agent_origin(self, agent_origin):
        self.agent.set(agent_origin) 

    """
    MDP Logic
    """

    # perform one step in an episode
    def step(self, action):
        # perform action on environment and update state
        self.state["agent"] = self.agent.move(action)
        self.state["info"] = self.get_info()
        self.state["target"][2] = atan2(self.state["target"][1]-self.state["agent"][1],
                                        self.state["target"][0]-self.state["agent"][0])

        # update dist
        self.dist_min = min(self.dist_min, self.dist_to_goal())

        # update trajectory
        self.trajectory.append(self.state["agent"][:2] + self.agent.origin)

        # query if done
        done, done_state = self.is_done()

        # compute reward
        reward = self.compute_reward(action, done_state)

        # increment episode length
        self.ep_len += 1

        return self.state_to_obs(), reward, done, {}

    # done function
    def is_done(self):
        dist_current = self.dist_to_goal()

        # reached goal
        if dist_current < 0.03:
            print("GOAL")
            return True, 0

        # passed goal
        if np.dot(self.state["agent"][:2], self.state["target"][:2]) > np.linalg.norm(self.state["target"][:2]) ** 2:
            print("PASSED GOAL")
            return True, 1

        # episode ends (comment out on training)
        if self.ep_len > 5000:
            return True, 3

        # out of canvas
        if self.state["agent"][0] > 1 or self.state["agent"][0] < -1 or self.state["agent"][1] > 1 or self.state["agent"][1] < -1:
            return True, 2
        
        return False, -1
    
    # reward function
    def compute_reward(self, action, done_state):
        rot_diff = self.state["info"][2] - self.state["agent"][2]
        if rot_diff > np.pi:
            rot_diff -= 2 * np.pi
        elif rot_diff < -np.pi:
            rot_diff += 2 * np.pi
        rot_diff = round(rot_diff, 2)
        rot_reward = []
        for i in range(5):
            angle = -np.pi / 4 + rot_diff + (np.pi / 8 * i) + np.pi / 2
            tr = 1 - 4 * fabs(0.5 - modf(0.25 + 0.5 * angle % (2 * np.pi) / np.pi)[0])
            rot_reward.append(tr)

        dist_current = self.dist_to_goal()
        dist_reward = 2 ** (dist_current / self.dist_init)

        reward = round(rot_reward[action] * 5, 2) * dist_reward - 5
        #print(f"dist_reward:{dist_reward:4f}, rot_reward:{rot_reward[action]:4f}, reward:{reward:4f}")

        if done_state == 0:
            reward = 200
            self.agent.stop()
        elif done_state == 1:
            reward = -100
            self.agent.stop()
        elif done_state == 2:
            reward = -200
            self.agent.stop()
        elif done_state == 3:
            reward = -200
            self.agent.stop()

        return reward

    # reset the environment
    def reset(self):
        self.state["agent"] = self.agent.reset()
        self.state["info"] = self.get_info()

        self.ep_len = 0

        return self.state_to_obs()

    # set the target coordinates
    def set(self, target_pos):
        rot = atan2(target_pos[1] - self.state["agent"][1], target_pos[0] - self.state["agent"][0])
        self.state["target"] = np.append(target_pos, rot)
        self.targets.append(self.state["target"][:2] + self.agent.origin)

        self.dist_init = self.dist_to_goal()
        self.dist_min = self.dist_init

    # stop the agent
    def stop(self):
        self.state["info"] = self.get_info()

    # render the trajectory
    def show(self):
        plt.scatter(*zip(*self.trajectory), s=0.3)
        plt.scatter(*zip(*self.targets), c='red',s=0.3)
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.show()

    """
    HELPER FUNCTIONS
    """

    # state (dict) to observation (np array)
    def state_to_obs(self):
        return np.concatenate((self.state["agent"], self.state["target"]))

    # return updated (recomputed) info
    def get_info(self):
        rot = atan2(self.state["target"][1] - self.state["agent"][1], self.state["target"][0] - self.state["agent"][0])
        return [ self.dist_to_goal(), self.similarity_to_goal(), rot ]

    # returns the distance between the current agent and the goal
    def dist_to_goal(self):
        return np.linalg.norm(self.state["target"][:2] - self.state["agent"][:2])
    
    # returns the cosine similarity between the current agent and the goal
    def similarity_to_goal(self):
        va = self.state["agent"][:2]
        vt = self.state["target"][:2]
        if np.allclose(va, np.zeros(2)):
            rot = self.state["agent"][2]
            va = np.array([np.cos(rot), np.sin(rot)])
        return np.dot(va, vt) / (np.linalg.norm(va) * np.linalg.norm(vt))
 
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
    def __init__(self, n_angular):
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

        # initialize parameters
        self.model_name = "turtlebot3_burger"
        self.orientations = np.array([i * np.pi / 4 for i in range(8)])
        self.origin = np.array([ 0., 0. ])
        self.n_angular = n_angular
        self.max_angular_vel = 1.5
        self.linear_x = 0
        self.angular_z = 0

    # issue a move command to the turtlebot and returns its updated position
    def move(self, action):
        cmd = Twist()
        #cmd.linear.x = ((self.n_linear - 1) / 2 - action[0]) * self.max_linear_vel
        #cmd.angular.z = ((self.n_angular - 1) / 2 - action[1]) * self.max_angular_vel
        cmd.angular.z = ((self.n_angular - 1) / 2 - action) * self.max_angular_vel
        if cmd.angular.z != 0:
            cmd.linear.x = 0
        else:
            cmd.linear.x = 0.15
        self.linear_x = cmd.linear.x
        self.angular_z = cmd.angular.z
        self.cmd_vel.publish(cmd)

        return self.get_state()

    # stop the agent
    def stop(self):
        self.cmd_vel.publish(Twist())

    # resets the turtlebot in the origin with a random orientation
    def reset(self):
        # reset the origin
        self.origin += self.get_state()[:2]
        
        # turtlebot has stopped
        self.linaer_x = 0
        self.angular_z = 0

        return self.get_state()

    def set(self, agent_origin):
        self.origin = agent_origin

        cmd = ModelState()
        cmd.model_name = self.model_name

        cmd.pose.position.x = agent_origin[0]
        cmd.pose.position.y = agent_origin[1]

        orientation = quaternion_from_euler(0, 0, np.random.choice(self.orientations))
        cmd.pose.orientation.x = orientation[0]
        cmd.pose.orientation.y = orientation[1]
        cmd.pose.orientation.z = orientation[2]
        cmd.pose.orientation.w = orientation[3]
        self.set_model_state(cmd)

    def get_state(self):
        resp = self.get_model_state(self.model_name, "")
        x = resp.pose.position.x - self.origin[0]
        y = resp.pose.position.y - self.origin[1]
        rot = euler_from_quaternion([resp.pose.orientation.x, resp.pose.orientation.y, resp.pose.orientation.z, resp.pose.orientation.w])

        return np.array([ x, y, rot[2], self.linear_x, self.angular_z ])
