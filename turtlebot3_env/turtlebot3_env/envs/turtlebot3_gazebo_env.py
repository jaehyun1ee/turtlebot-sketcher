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
        launchfile = str(os.getcwd()) + "/turtlebot3_simulations/turtlebot3_gazebo/launch/turtlebot3_empty_world.launch"         
        os.environ["TURTLEBOT3_MODEL"] = "burger"
        subprocess.Popen([sys.executable, os.path.join(ros_path, b"roslaunch"), "-p", "11311", launchfile])
        print ("Gazebo launched!")

        # initialize agent
        self.agent = Agent()

        # track
        self.track_width = 0.01
        self.track_length = 0
        self.track_unit = np.array([0, 0])
        self.track_ortho_unit = np.array([0, 0])
        self.track_tile_num = 10
        self.track_tile_visited = np.array([False for _ in range(self.track_tile_num)])

        # initialize environment (MDP definition)
        self.observation_space = spaces.Box(low=np.array([-1, -1, -np.pi, -1, -1, 0, -1]), high=np.array([1, 1, np.pi, 1, 1, 4, 1]), shape=(7,), dtype=np.float64) # (x_agent, y_agent, rot_agent, x_goal, y_goal, dist_to_goal, similarity_to_goal)
        self.action_space = spaces.Box(low=np.array([0, -1.5]), high=np.array([1.5, 1.5]), shape=(2,), dtype=np.float32) # linear-x, and angular-z
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
        cmd = Twist()
        cmd.linear.x = action[0]
        cmd.angular.z = action[1]
        # update state
        self.state["agent"] = self.agent.move(cmd)
        self.state["info"] = self.get_info()

        # compute tile index
        track_tile_idx = self.get_track_tile()
        
        # compute reward
        reward = self.compute_reward(track_tile_idx)

        # query if the episode is done
        done = self.is_done(track_tile_idx)

        return self.state_to_obs(), reward, done, {}
    
    # reward function
    def compute_reward(self, track_tile_idx):
        # living cost
        reward = -0.1

        # out of track
        if track_tile_idx < 0:
            reward -= 100
        # newly visited tile
        elif not self.track_tile_visited[track_tile_idx]:
            self.track_tile_visited[track_tile_idx] = True
            reward += 1000 / self.track_tile_num

        return reward 

    # returns true if the episode is done
    def is_done(self, track_tile_idx):
        if track_tile_idx < 0:
            return True
        d = self.dist_to_goal()
        return d < 0.01

    # reset the environment
    def reset(self):
        self.state["agent"] = self.agent.reset()
        self.state["target"] = self.random_vector()
        self.state["info"] = self.get_info()
        self.set_track()

        return self.state_to_obs()

    # stop the agent
    def stop(self):
        self.state["agent"] = self.agent.move(Twist())
        self.state["info"] = self.get_info()

    """
    HELPER FUNCTIONS
    """

    # state (dict) to observation (np array)
    def state_to_obs(self):
        return np.concatenate([self.state["agent"], self.state["target"], self.state["info"]], dtype=np.float64)

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

    # returns a random 2D vector in (-1, 1) x (-1, 1)
    def random_vector(self):
        rand = np.zeros(2)
        while np.allclose(rand, np.zeros(2)):
            rand = np.random.rand(2) * 2 - 1
        return rand

    # 
    def set_track(self):
        self.track_length = np.linalg.norm(self.state["target"])
        self.track_unit = self.state["target"] / self.track_length
        self.track_ortho_unit = np.array([ -self.track_unit[1], self.track_unit[0] ])
        self.track_tile_visited = np.array([False for _ in range(self.track_tile_num)])

    # returns the tile number where the agent is located at, and returns -1 if out of track
    def get_track_tile(self):
        # project agent's position into track
        a = np.dot(self.state["agent"][:2], self.track_unit)
        b = np.dot(self.state["agent"][:2], self.track_ortho_unit)

        # agent is out of track
        if not (a >= 0 and a <= self.track_length and b >= -self.track_width and b <= self.track_width):
            return -1

        return int(a * self.track_tile_num / self.track_length)
    
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
        self.rate = rospy.Rate(10)
        self.model_name = "turtlebot3_burger"

    # issue a move command to the turtlebot and returns its updated position
    def move(self, cmd):
        self.cmd_vel.publish(cmd)
        self.rate.sleep()

        return self.get_state()

    # resets the turtlebot in the origin with a random orientation
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
