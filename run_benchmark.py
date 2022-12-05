import numpy as np
import os, site

import stroker.STROKER_INTERFACE as stroker
import commander.COMMANDER_INTERFACE as commander

"""
Utilities to link Commander and Stroker
"""

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

"""
Run benchmarks
"""

if __name__ == '__main__':
    PATH_PREFIX = "./benchmark/Drawing_"
    PATH_DIRNAME = [ "25_bench", "51_apple", "95_candle", "97_bucket" ] 
    PATH_POSTFIX_RAW = "/raw.npy"
    PATH_POSTFIX_PROCESSED = "/strokes.npy"
    PATH_POSTFIX_COMMANDER = "/commander.npy"
    
    # make environment
    env, model = stroker.make_env()

    # load model
    try:
        for dirname in PATH_DIRNAME:
            # RUN COMMANDER
            print("RUNNING COMMANDER FOR " + dirname)
            path_raw = PATH_PREFIX + dirname + PATH_POSTFIX_RAW
            strokes_raw = commander.get_img_strokes(path_raw)

            strokes = []
            commands = []
            for i in range(len(strokes_raw)):
                # process raw into bitmap
                stroke = strokes_raw[i]
                img = commander.stroke2img(stroke)
                start = [ stroke[0][0], stroke[1][0] ]
                end = [ stroke[0][-1], stroke[1][-1] ]

                # for stroker-only
                strokes.append(commander.stroke2points(stroke))

                # commander output
                commands.append(commander.img2points(img, start, end))

            # commander-only
            commander.visualize_commander(commands, PATH_PREFIX + dirname + "/commander.png")

            # RUN STROKER
            print("RUNNING STROKER FOR " + dirname)
            
            # stroker-only
            for stroke in strokes:
                start, vectors = commands_to_vectors(stroke)
                start, strokes = vectors_to_strokes(start, vectors)

                stroker.draw(env, model, start, strokes)
            stroker.visualize_stroker(env, PATH_PREFIX + dirname + "/stroker.png")
            stroker.clear(env)

            # commander-and-stroker
            for command in commands:
                start, vectors = commands_to_vectors(command)
                start, strokes = vectors_to_strokes(start, vectors)

                stroker.draw(env, model, start, strokes)
            stroker.visualize_stroker(env, PATH_PREFIX + dirname + "/commander_and_stroker.png")
            stroker.clear(env)
    finally:
        stroker.close_env(env)
