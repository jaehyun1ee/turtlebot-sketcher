import gym
import numpy as np
import turtlebot3_env
import os, site

os.environ["LD_PRELOAD"] = site.USER_SITE + "/torch/lib/../../torch.libs/libgomp-d22c30c5.so.1.0.0"

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from callback import TensorboardCallback

if __name__ == '__main__':
    # make environment
    env = gym.make('turtlebot3_env/Turtlebot3-v0')
    #env = DummyVecEnv([lambda: gym.make('turtlebot3_env/Turtlebot3-v0')])
    env = Monitor(env, "./logs/", info_keywords=("is_success","pixel_diff"))

    try:
        try:
            model = DQN.load("dqn_turtlebot", env=env, tensorboard_log="./logs/")
            model.set_parameters("dqn_turtlebot")
            print("model loaded")
        except:
            model = DQN("MlpPolicy", env, verbose=1, tensorboard_log="./logs/")
            print("model load failed")
        
        model.learn(total_timesteps=1000000, log_interval=50, reset_num_timesteps=False)
        print("training complete")
        model.save("dqn_turtlebot")
    finally:
        env.close()
