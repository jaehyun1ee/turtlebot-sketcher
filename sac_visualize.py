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
        
        while True:
            rewards = []
            state = env.reset()
            while True:
                action, _ = model.predict(state, deterministic=True)
                state, reward, done, _ = env.step(action)
                print(f"current position: {state[0]:5f}, {state[1]:5f}")
                print(f"target position: {state[3]:5f}, {state[4]:5f}")
                print(f"reward: {reward}")
                print(f"done: {done}")
                rewards.append(reward)
                if done:
                    state = env.reset()
                    break
            print(f"total rewards: {rewards}")
            env.stop()
            k = int(input("ENTER 1 TO CONTINUE"))
            if k != 1:
                break
        input("WAIT TO CLOSE")
    finally:
        env.close()


