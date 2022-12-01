import numpy as np
import os, site

#import commander.COMMANDER_INTERFACE as commander

import gym
import turtlebot3_env
from stable_baselines3 import DQN

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

def commands_to_vectors(commands):
    start = [ commands[0][0], 127 - commands[0][1] ]

    vectors = []
    for i in range(len(commands) - 1):
        vectors.append([ commands[i + 1][0] - commands[i][0], commands[i][1] - commands[i + 1][1] ])

    return start, vectors

def vectors_to_strokes(start, vectors):
    start = [ start[0] / 64 - 1, start[1] / 64 - 1 ]

    strokes = []
    for v in vectors:
        strokes.append([ v[0] / 64, v[1] / 64 ])

    return start, strokes

if __name__ == "__main__":
    PATH_PREFIX = "./benchmark/Drawing_"
    PATH_DIRNAME = [ "25_bench", "51_apple", "95_candle", "97_bucket" ] 
    PATH_POSTFIX_COMMANDER = "/commander.npy"
    PATH_POSTFIX_ORIGINAL = "/original.npy"

    # make environment
    env = gym.make('turtlebot3_env/Turtlebot3-real-v0')

    # load model
    try:
        model = DQN("MlpPolicy", env, verbose=1)
        model = DQN.load("./dqn_turtlebot")
        model.set_parameters("./dqn_turtlebot")

        for dirname in PATH_DIRNAME:
            path_command = PATH_PREFIX + dirname + PATH_POSTFIX_COMMANDER
            commands = np.load(path_command, allow_pickle=True)

            for command in commands:
                start, vectors = commands_to_vectors(command)
                start, strokes = vectors_to_strokes(start, vectors)

                draw(env, model,start, strokes)
            print("commander with stroker : " + dirname)
            env.show(PATH_PREFIX + dirname + "/commander_and_stroker.png")
            env.clear()
            
            path_original = PATH_PREFIX + dirname + PATH_POSTFIX_ORIGINAL
            originals = np.load(path_original, allow_pickle=True)
            for original in originals:
                start, vectors = commands_to_vectors(original)
                start, strokes = vectors_to_strokes(start, vectors)

                draw(env, model, start, strokes)
            print("stroker alone : " + dirname)
            env.show(PATH_PREFIX + dirname + "/stroker.png")
            env.clear()
    finally:
        env.close()
