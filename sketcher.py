import numpy as np
import os, site

#import commander.COMMANDER_INTERFACE as commander

import gym
import turtlebot3_env
from stable_baselines3 import DQN

def draw(env, model, title, start, strokes):
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
    env.show(title)
    #input("ENTER TO PROCEED TO NEXT EXAMPLE")

def commands_to_vectors(commands):
    start = commands[0]

    vectors = []
    for i in range(len(commands) - 1):
        vectors.append([ commands[i + 1][0] - commands[i][0], commands[i + 1][1] - commands[i][1] ])

    return start, vectors

def vectors_to_strokes(start, vectors):
    start = [ start[0] / 64 - 1, start[1] / 64 - 1 ]

    strokes = []
    for v in vectors:
        strokes.append([ v[0] / 64, v[1] / 64 ])

    return start, strokes

if __name__ == "__main__":
    PATH_PREFIX = "./benchmark/stroke_process_"
    PATH_DIRNAME = [ "15_campfire", "37_calculator", "52_bandage", "88_cannon", "96_bicycle", "35_birthdaycake", "43_brain", "6_effel_tower", "95_candle", "97_bucket" ]
    PATH_POSTFIX = "/stroker_input.npy"

    # make environment
    env = gym.make('turtlebot3_env/Turtlebot3-real-v0')

    # load model
    try:
        model = DQN("MlpPolicy", env, verbose=1)
        model = DQN.load("./stroker/dqn_turtlebot")
        model.set_parameters("./stroker/dqn_turtlebot")

        for dirname in PATH_DIRNAME:
            path = PATH_PREFIX + dirname + PATH_POSTFIX
            commands = np.load(path)
            start, vectors = commands_to_vectors(commands)
            start, strokes = vectors_to_strokes(start, vectors)

            draw(env, model, dirname, start, strokes)
            env.clear()
    finally:
        env.close()
