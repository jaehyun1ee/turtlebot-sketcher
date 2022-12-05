from COMMANDER import * 
from COMMANDER_INTERFACE import *


def extract_drawing(name, path_num, draw_num):
    path = drawings_path[path_num]
    with open(path) as f:
        print("--- Loading file: {}".format(f))
        drawings = json.load(f)
        
    single_drawing = drawings[draw_num]
    plot_bitmap(stroke_to_bitmap(single_drawing), "original")
    np.save("../output/sample_drawing_{}_{}_{}".format(name, path_num, draw_num), single_drawing)

def extract():
    extract_drawing("bench", 25, 88)
    extract_drawing("apple", 51, 40)
    extract_drawing("candle", 95, 81)
    extract_drawing("bucket", 97, 0)

extract()