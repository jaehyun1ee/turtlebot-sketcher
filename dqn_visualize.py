import gym
import numpy as np
import turtlebot3_env
import os, site

os.environ["LD_PRELOAD"] = site.USER_SITE + "/torch/lib/../../torch.libs/libgomp-d22c30c5.so.1.0.0"

from stable_baselines3 import DQN

if __name__ == '__main__':
    # make environment
    env = gym.make('turtlebot3_env/Turtlebot3-v0')

    try:
        model = DQN("MlpPolicy", env, verbose=1)
        model = DQN.load("dqn_turtlebot")
        model.set_parameters("dqn_turtlebot")
        
        while True:
            rewards = []
            state = env.reset()
            while True:
                action, _ = model.predict(state, deterministic=True)
                state, reward, done, _ = env.step(action)
                rewards.append(reward)
                if done:
                    print(f"current position: {state[0]:5f}, {state[1]:5f}, {state[2]:5f}")
                    print(f"target position: {state[3]:5f}, {state[4]:5f}, {state[5]:5f}")
                    print(f"reward: {sum(rewards)}")
                    print(f"done: {done}")
                    state = env.reset()
                    break
            env.stop()
            k = int(input("ENTER 1 TO CONTINUE"))
            if k != 1:
                break
        input("WAIT TO CLOSE")
    finally:
        env.close()


