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
        model = SAC.load("sac_turtlebot")
        
        rewards = []
        state = env.reset()
        while True:
            action, _ = model.predict(state, deterministic=True)
            state, reward, done, _ = env.step(action)
            print(state, reward, done)
            rewards.append(reward)
            if done:
                state = env.reset()
                break
        print(rewards)
    finally:
        env.close()


