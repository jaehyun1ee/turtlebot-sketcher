import gym
import turtlebot3_env
import os, site

os.environ["LD_PRELOAD"] = site.USER_SITE + "/torch/lib/../../torch.libs/libgomp-d22c30c5.so.1.0.0"

from stable_baselines3.common.env_checker import check_env

if __name__ == '__main__':
    # make environment
    env = gym.make('turtlebot3_env/Turtlebot3-v0')
    try:
        check_env(env)
    finally:
        env.close()

