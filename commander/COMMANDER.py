import concurrent.futures, random, json
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import rdp, bresenham
from tqdm import tqdm
import glob

dir_path = "../data/processed_data/*"
drawings_path = glob.glob(dir_path)

WIDTH, HEIGHT = 128, 128
FLOAT_ERROR = 1e-6
assert WIDTH == HEIGHT

##########################################################################

# axis 표기법
# x_axis = [1, 2, 3, 4, 5, 6]
# y_axis = [2, 4, 6, 8, 10, 12]

# points 표기법
# points = [[1, 2], [2, 4], [3, 6], [4, 8], [5, 10], [6, 12]]

# 두 표기법 간의 변환함수
def axis_to_points(x_axis, y_axis):
    return np.array([x_axis, y_axis]).T.reshape(-1, 2)

def one_axis_to_points(xy_axis):
    return np.array([xy_axis[0], xy_axis[1]]).T.reshape(-1, 2).tolist()

def points_to_axis(points):
    points = np.array(points)
    return points[:, 0], points[:, 1]
    
def cord_to_index(x, y):
    # convert (x, y) -> (i, j)
    # x=0~1 -> j=0          ; j = np.floor(x)
    # y=0~1 -> i=(HEIGHT-1) ; i = np.floor(HEIGHT-y)
    assert 0 <= x < WIDTH
    assert 0 <= y < HEIGHT
    if y == 0:
        y = 1
    return int(HEIGHT-y), int(x)

def line_augment(bitmap, start, end):
    xs, ys = start
    xe, ye = end
    xs, ys, xe, ye = map(int, [xs, ys, xe, ye])
    for point in bresenham.bresenham(xs, ys, xe, ye):
        xp, yp = point
        bitmap[cord_to_index(xp, yp)] = 1
    return

def stroke_to_bitmap(strokes, bitmap=None):
    # function to convert vector-format strokes
    # to the bitmap numpy array
    
    if bitmap is None:
        bitmap = np.zeros(shape=(HEIGHT, WIDTH), dtype=bool)
    
    for stroke in strokes: 
        x_axis, y_axis = stroke
        points = axis_to_points(x_axis, y_axis)
        for start, end in zip(points[:-1], points[1:]):
            # line[start -> end]
            # where start = (xi, yi) / end = (xj, yj)
            if float("NaN") in start or float("NaN") in end:
                print(points)
            line_augment(bitmap, start, end)
    
    return bitmap

def point_to_bitmap(point, bitmap=None):
    
    # function to convert vector-format strokes
    # to the bitmap numpy array
    
    if bitmap is None:
        bitmap = np.zeros(shape=(HEIGHT, WIDTH), dtype = bool)
    x, y = map(int, point)
    bitmap[cord_to_index(x,y)] = 1  
 
    return bitmap

def plot_bitmap(bitmap, path):
    bitmap = np.flip(bitmap, 0)
    bitmap = bitmap.astype(float)
    if len(bitmap.shape) == 2:
        plt.imshow(bitmap, cmap='Greys')
    elif len(bitmap.shape) == 3:
        plt.imshow(bitmap)
    plt.show()
    plt.savefig(path)

import traceback

def stroke_to_trainset(strokes):
    mult_input_data, mult_output_data = [], []
    
    for rand_k in [np.random.randint(len(strokes))]:
        stroke_k = strokes[rand_k]
        x_axis, y_axis = stroke_k
        # [x0, x1, x2, ..., xm]

        for random_l in [np.random.randint(len(x_axis))]:
            prev_stroke = np.array([x_axis[:random_l+1], y_axis[:random_l+1]])
            end_point = [x_axis[random_l], y_axis[random_l]]

            if random_l + 1 == len(x_axis):
                next_point = [x_axis[-1], y_axis[-1]]
            else:
                next_point = [x_axis[random_l+1], y_axis[random_l+1]]
                
            error_epsilon = 4
            stroke_start = prev_stroke[0]
            prev_stroke = np.clip(
                prev_stroke + error_epsilon * (np.random.random(prev_stroke.shape) * 2 - 1),
                0, WIDTH-1
            )
            prev_stroke[0] = stroke_start
        
            bitmap_img = stroke_to_bitmap([stroke_k])
            bitmap_prev = stroke_to_bitmap([prev_stroke])
            bitmap_endpoint = point_to_bitmap(end_point)
            bitmap_nextpoint = next_point

            input_data = np.array([bitmap_img, bitmap_prev, bitmap_endpoint], dtype = bool) # (channel, height, width) order
            input_data = np.moveaxis(input_data, 0, -1) # (height, width, channel) order -> suitable for Tensorflow
            output_data = np.array(bitmap_nextpoint, dtype = np.int8)
            
            mult_input_data.append(input_data)
            mult_output_data.append(output_data)
        
    return mult_input_data, mult_output_data

def prepare_data():
    while True:
        mode = input("Do you want to use processed .npz file instead of reading from /data? (y/n) ")
        if mode in ['y', 'Y']:
            CALCULATE_MODE = False
        elif mode in ['n', 'N']:
            CALCULATE_MODE = True
        else:
            continue
        break
    return CALCULATE_MODE

import tensorflow as tf
#import tensorflow_addons as tfa
import keras.api._v2.keras as keras
from keras import layers

def prepare_gpu():
    gpu_list = tf.config.list_physical_devices('GPU')

    print("Num GPUs Available: ", len(gpu_list))
    for gpu in gpu_list:
        print("---- gpu ----", gpu)

# 모델 설계

def batchnormal(name='unnamed'):
    # (batch, channel, width, height) -> axis = 1
    return layers.BatchNormalization(axis=1, name=name) 

def relu():
    return layers.Activation('relu')

class IdentityBlock:
    def __init__(self, filters, name='unnamed'):
        filter1, filter2 = filters

        self.compress1 = layers.Conv2D(filter1, (3, 3), kernel_initializer='he_normal', padding='same',
            name='{}-identity1'.format(name)
        )
        
        self.compress2 = layers.Conv2D(filter2, (3, 3), kernel_initializer='he_normal', padding='same',
            name='{}-identity2'.format(name)
        )

        self.batch1 = batchnormal('{}-idnorm1'.format(name))
        self.batch2 = batchnormal('{}-idnorm2'.format(name))
        self.active = relu()
    
    def __call__(self, inputs):
        x = inputs
        x = self.active(self.batch1(self.compress1(x)))
        x = self.active(self.batch2(self.compress2(x)) + inputs)
        return x

class ConvBlock:
    def __init__(self, filters, strides=(1,1), name='unnamed'):
        filter1, filter2 = filters

        self.compress1 = layers.Conv2D(filter1, (3, 3), kernel_initializer='he_normal', padding='same', strides=strides,
            name = '{}-conv1'.format(name)
        )

        self.compress2 = layers.Conv2D(filter2, (3, 3), kernel_initializer='he_normal', padding='same',
            name = '{}-conv2'.format(name)
        )

        self.shortcut = layers.Conv2D(filter2, (1, 1), kernel_initializer='he_normal', strides=strides,
            name = '{}-short'.format(name)
        )
        
        self.batch1 = batchnormal('{}-convnorm1'.format(name))
        self.batch2 = batchnormal('{}-convnorm2'.format(name))
        self.active = relu()
    
    def __call__(self, inputs):
        x = inputs
        x = self.active(self.batch1(self.compress1(x)))
        x = self.active(self.batch2(self.compress2(x)) + self.shortcut(inputs))
        return x

def Resnet():
        # 128 x 128 @ 3 || width x height @ channel
    #
    # Output ->
    # 128 x 128 @ 1 || width x height @ channel

    custom_layers = [
        layers.Conv2D(16, (5,5), strides=(1,1), padding='same',
            kernel_initializer='he_normal', name='first-layer'), # 128 x 128 @ 16
        batchnormal('first-batch'), relu(),
        
        IdentityBlock(filters=(16,16), name = '16c-first'),   # 128 x 128 @ 16
        IdentityBlock(filters=(16,16), name = '16c-second'),  # 128 x 128 @ 16
        # IdentityBlock(filters=(16,16), name = '16c-third'),   # 128 x 128 @ 16

        ConvBlock(filters=(32,32), strides=(2,2), name = '32c-first'),  # 64 x 64 @ 32
        IdentityBlock(filters=(32,32), name = '32c-second'),  # 64 x 64 @ 32
        # IdentityBlock(filters=(32,32), name = '32c-third'),   # 64 x 64 @ 32

        ConvBlock(filters=(64,64), strides=(2,2), name = '64c-first'),  # 32 x 32 @ 64
        IdentityBlock(filters=(64,64), name = '64c-second'),  # 32 x 32 @ 64
        # IdentityBlock(filters=(64,64), name = '64c-third'),   # 32 x 32 @ 64

        ConvBlock(filters=(64,64), strides=(2,2), name = '64cc-first'),  # 16 x 16 @ 64
        IdentityBlock(filters=(64,64), name = '64cc-second'),  # 16 x 16 @ 64
        # IdentityBlock(filters=(64,64), name = '64cc-third'),   # 16 x 16 @ 64

        ConvBlock(filters=(64,64), strides=(2,2), name = '64ccc-first'),  # 8 x 8 @ 64
        IdentityBlock(filters=(64,64), name = '64ccc-second'),  # 8 x 8 @ 64
        # IdentityBlock(filters=(64,64), name = '64ccc-third'),   # 8 x 8 @ 64

        ConvBlock(filters=(128,128), strides=(2,2), name = '128c-first'),  # 4 x 4 @ 128
        IdentityBlock(filters=(128,128), name = '128c-second'),  # 4 x 4 @ 128
        # IdentityBlock(filters=(128,128), name = '128c-third'),   # 4 x 4 @ 128

        layers.GlobalAveragePooling2D(),
        layers.Dense(2),
        layers.Activation(tf.nn.sigmoid)
    ]

    input_layer = layers.Input(shape=(HEIGHT,WIDTH,3))
    x = input_layer
    for layer in custom_layers:
        x = layer(x)
    output_layer = x * WIDTH

    return keras.Model(inputs=input_layer, outputs=output_layer)


def load_trainset(CALCULATE_MODE):
    def train_gen():
        for idx in range(train_size):
            yield train_input[idx], train_output[idx]

    def val_gen():
        for idx in range(valid_size):
            yield val_input[idx], val_output[idx] 

    print("\nLoading trainset...")

    # random choose data1
    train_ratio = 0.95
    input_dataset, output_dataset = [], []


    if CALCULATE_MODE:
        print("Processing from /data...")
        for path_single in drawings_path[4:5]:
            with open(path_single) as f:
                print("--- Loading file: {}".format(f))
                drawings = json.load(f)
            with concurrent.futures.ProcessPoolExecutor() as executor:
                for mult_input_data, mult_output_data in tqdm(executor.map(stroke_to_trainset, drawings), total=len(drawings)):
                    for input_data, output_data in zip(mult_input_data, mult_output_data):
                        input_dataset.append(input_data)
                        output_dataset.append(output_data.flatten())

        print("Saving processed results into .npz format...")
        np.save('../output/input_dataset', input_dataset)
        np.save('../output/output_dataset', output_dataset)
    else:
        print("Loading processed cache...")
        input_dataset = np.load('../output/input_dataset.npy')
        output_dataset = np.load('../output/output_dataset.npy')
        
    idx = np.random.randint(len(input_dataset))
    plot_bitmap(input_dataset[idx], "sample_in")
    plot_bitmap(output_dataset[idx], "sample_out")
        
    train_size = int(len(input_dataset) * train_ratio)
    train_input = input_dataset[:train_size]
    train_output = output_dataset[:train_size]
    val_input = input_dataset[train_size:]
    val_output = output_dataset[train_size:]

    valid_size = len(val_input)

    print("Conversion [list -> tf.data.Dataset] on progress...")

    train_dataset = tf.data.Dataset.from_generator(
        train_gen, output_types=(bool, np.int8), output_shapes=([HEIGHT, WIDTH, 3], [2])
    ).batch(32)

    val_dataset = tf.data.Dataset.from_generator(
        val_gen, output_types=(bool, np.int8), output_shapes=([HEIGHT, WIDTH, 3], [2])
    ).batch(32)

    print("****** TRAINING INFORMATIONS ******")
    print("Train size:", train_size)
    print("Valid size:", valid_size)
    print("***********************************")
    
    return train_dataset, val_dataset

############################ !!! 직접 짠 코드 아님 !!! 구글에서 긁어옴 !!! ############################

def vis(history,name):
    plt.title(f"{name.upper()}")
    plt.xlabel('epochs')
    plt.ylabel(f"{name.lower()}")
    value = history.history.get(name)
    val_value = history.history.get(f"val_{name}",None)
    epochs = range(1, len(value)+1)
    plt.plot(epochs, value, 'b-', label=f'training {name}')
    if val_value is not None :
        plt.plot(epochs, val_value, 'r:', label=f'validation {name}')
    plt.legend(loc='upper center', bbox_to_anchor=(0.05, 1.2) , fontsize=10 , ncol=1)
    
def plot_history(history):
    key_value = list(set([i.split("val_")[-1] for i in list(history.history.keys())]))
    plt.figure(figsize=(12, 4))
    for idx , key in enumerate(key_value) :
        plt.subplot(1, len(key_value), idx+1)
        vis(history, key)
    plt.tight_layout()
    plt.savefig("../output/performance.png")

############################ !!! 직접 짠 코드 아님 !!! 구글에서 긁어옴 !!! ############################

# 학습 데이터를 가지고 모델을 훈련, epochs 횟수만큼 반복

def scheduler(epoch, _lr):
    if epoch <= 2:
        return 1e-4
    else:
        return 1e-5
lr_modify = keras.callbacks.LearningRateScheduler(scheduler)

def train():
    
    CALCULATE_MODE = prepare_data()
    prepare_gpu()
    train_dataset, val_dataset = load_trainset(CALCULATE_MODE)
    
    print("Preparing Model...")
    CIFARDNNModel = Resnet()
    """
    CIFARDNNModel.compile(
        optimizer=tfa.optimizers.AdamW(
            # learning_rate=2e-5,
            weight_decay=1e-6
        ),
        loss=keras.losses.MeanSquaredError(),
        metrics=['mean_absolute_error', 'mean_absolute_percentage_error']
    )
    """
    print("Model prepared!")
    CIFARDNNModel.summary()
    
    print("*** Ready to train ***")
    history = CIFARDNNModel.fit(
        train_dataset,
        epochs= 30, batch_size=32,
        validation_data=val_dataset,
        callbacks=[lr_modify]
    )

    print("visualizing...")
    plot_history(history)

    print("saving weights...")
    CIFARDNNModel.save_weights('./model/commander/checkpoint.mdl')
    print("goodbye")
    
if __name__ == "__main__":
    train()
