#!/usr/bin/env python
# coding: utf-8

# In[7]:


import concurrent.futures, random, json
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import rdp, bresenham
from tqdm import tqdm


# In[8]:


import os

raw_dir_path = "../data/raw_data/"
out_dir_path = "../data/processed_data/"
raw_drawings_path = os.listdir(raw_dir_path)


# In[9]:


WIDTH, HEIGHT = 128, 128
FLOAT_ERROR = 1e-6
assert WIDTH == HEIGHT

def axis_to_points(x_axis, y_axis):
    return np.array([x_axis, y_axis]).T.reshape(-1, 2)

def points_to_axis(points):
    points = np.array(points)
    return points[:, 0], points[:, 1]

def extract_minmax(L, min_L, max_L):
    return min(min(L), min_L), max(max(L), max_L)

def relax(minval, maxval):
    assert minval <= maxval
    dist = maxval - minval
    return minval-0.04*dist, maxval+0.04*dist


# In[10]:


def get_frame(strokes):
    min_x, min_y = float("+inf"), float("+inf")
    max_x, max_y = float("-inf"), float("-inf")
    for stroke in strokes:
        x_axis, y_axis = stroke
        min_x, max_x = extract_minmax(x_axis, min_x, max_x)
        min_y, max_y = extract_minmax(y_axis, min_y, max_y)
    min_x, max_x = relax(min_x, max_x)
    min_y, max_y = relax(min_y, max_y)
    return min_x, max_x, min_y, max_y


# In[11]:


def normalize(strokes):
    min_x, max_x, min_y, max_y = get_frame(strokes)
    if max_x - min_x < FLOAT_ERROR:
        xconv = lambda x: WIDTH // 2
    else:
        xconv = lambda x: (x - min_x) / (max_x - min_x) * WIDTH
    if max_y - min_y < FLOAT_ERROR:    
        yconv = lambda y: HEIGHT // 2
    else:
        yconv = lambda y: (y - min_y) / (max_y - min_y) * HEIGHT
    modified = []
    for stroke in strokes:
        x_axis, y_axis = stroke
        x_mod = list(map(xconv, x_axis))
        y_mod = list(map(yconv, y_axis))
        modified.append([x_mod, y_mod])
    return modified

def rdp_apply(strokes):
    modified = []
    for stroke in strokes:
        x_axis, y_axis, _ = stroke
        points = axis_to_points(x_axis, y_axis)
        mod_points = rdp.rdp(points, epsilon=2)
        x_mod, y_mod = points_to_axis(mod_points)
        modified.append([x_mod.tolist(), y_mod.tolist()])
    return modified

def preprocess(strokes):
    return normalize(rdp_apply(strokes))

# In[ ]:

for file_path in raw_drawings_path:
    processed = []
    print("Processing ", file_path)
    chunks = pd.read_json(raw_dir_path + file_path, lines = True, chunksize=10000)
    for chunk in chunks:
        raw_drawings = chunk['drawing']
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for drawing in tqdm(executor.map(preprocess, raw_drawings), total=len(raw_drawings)):
                processed.append(drawing)
    print(file_path, " len:", len(processed))
    with open(out_dir_path + file_path, 'w') as f:
        json.dump(processed, f)


# In[ ]:





# In[ ]:




