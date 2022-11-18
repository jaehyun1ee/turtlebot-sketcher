import gym
import turtlebot3_env
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

if __name__ == '__main__':
    # make environment
    env = gym.make('turtlebot3_env/Turtlebot3-v0')
    try:
        conti = True
        while conti:
            # reset environment
            env.reset()

            # a random policy
            done = False
            its = 0
            while not done:
                action = env.action_space.sample()
                next_state, reward, done, _ = env.step(action)
                print(f"it ({its});\nstate : {next_state}\naction : {action}\nreward : {reward}")
            env.stop()
            print(f"final : {env.state}")
            
            dist = sqrt(next_state[0] ** 2 + next_state[1] ** 2)
            print(f"action: {abs(action[1]/2):4f}\tdist: {dist:4f}\tdiff: {abs(abs(action[1]/2) - dist):4f}")

            # continue or not?
            key = int(input("ENTER 1 TO CONTINUE : "))
            if key != 1:
                conti = False

        # close environment
        input("ENTER TO CLOSE")
    finally:
        env.close()
