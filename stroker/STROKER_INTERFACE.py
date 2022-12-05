import numpy as np

import gym
import turtlebot3_env
from stable_baselines3 import DQN

def make_env():
    env = gym.make('turtlebot3_env/Turtlebot3-real-v0')
    model = DQN("MlpPolicy", env, verbose=1)
    model = DQN.load("./model/stroker")
    model.set_parameters("./model/stroker")

    return env, model

def draw(env, model, start, strokes):
    env.init_agent(start)
    for stroke in strokes:
        state = env.reset()
        env.set(stroke)
        while True:
            action, _ = model.predict(state, deterministic=True)
            state, reward, done, _ = env.step(action)
            if done:
                state = env.reset()
                break
        env.stop()

def visualize_stroker(env, path):
    env.show(path)

def clear(env):
    env.clear()

def close_env(env):
    env.close()
