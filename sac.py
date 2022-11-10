import gym
import numpy as np
import turtlebot3_env
import os, site

os.environ["LD_PRELOAD"] = site.USER_SITE + "/torch/lib/../../torch.libs/libgomp-d22c30c5.so.1.0.0"

from stable_baselines3 import SAC

if __name__ == '__main__':
    # make environment
    env = gym.make('turtlebot3_env/Turtlebot3-v0')

    try:
        model = SAC("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=10000, log_interval=100)
        print("training complete")
        model.save("sac_turtlebot")
    finally:
        env.close()


