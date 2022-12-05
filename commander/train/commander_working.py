





IS_RUNNING_ON_COLAB = None
 






IS_RUNNING_ON_COLAB = False






import concurrent.futures, random, json
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import rdp, bresenham
from tqdm import tqdm







if IS_RUNNING_ON_COLAB:
    df = pd.read_json("/content/drive/MyDrive/AI_Intro/full_raw_ambulance.ndjson", lines=True)
else:
    df = pd.read_json("../data/ambulance.ndjson", lines=True)
print(df.columns)















def axis_to_points(x_axis, y_axis):
    return np.array([x_axis, y_axis]).T.reshape(-1, 2)

def points_to_axis(points):
    points = np.array(points)
    return points[:, 0], points[:, 1]




from matplotlib.collections import LineCollection

def extract_minmax(L, min_L, max_L):
    return min(min(L), min_L), max(max(L), max_L)

def relax(minval, maxval):
    assert minval <= maxval
    dist = maxval - minval
    return minval-0.04*dist, maxval+0.04*dist

def get_frame(strokes):
    min_x, min_y = float("+inf"), float("+inf")
    max_x, max_y = float("-inf"), float("-inf")

    for stroke in strokes:
        x_axis, y_axis, _ = stroke
    
        
        min_x, max_x = extract_minmax(x_axis, min_x, max_x)
        min_y, max_y = extract_minmax(y_axis, min_y, max_y)

    min_x, max_x = relax(min_x, max_x)
    min_y, max_y = relax(min_y, max_y)

    return min_x, max_x, min_y, max_y

def plot_stroke(strokes):
    fig, ax = plt.subplots()
    
    min_x, max_x, min_y, max_y = get_frame(strokes)
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

    for stroke in strokes:
        
        x_axis, y_axis, _ = stroke
        points = axis_to_points(x_axis, y_axis).reshape(-1, 1, 2)
        segs   = np.concatenate([points[:-1], points[1:]], axis=1)
        cols = LineCollection(segs)
        cols.set_linewidth(1)
        ax.add_collection(cols)
    
    plt.show()

sample = df['drawing'][2+327-4]   
plot_stroke(sample)







def rdp_apply(strokes):
    modified = []
    for stroke in strokes:
        x_axis, y_axis, _ = stroke
        points = axis_to_points(x_axis, y_axis)
        mod_points = rdp.rdp(points, epsilon=2)
        x_mod, y_mod = points_to_axis(mod_points)
        modified.append([x_mod.tolist(), y_mod.tolist(), None])

    return modified

plot_stroke(sample)
print("", "RDP NOT Applied", "\n", "="*50, "\n", "RDP Applied")
plot_stroke(rdp_apply(sample))























with open('../data/processed_strokes_set.json') as f:
    strokes_set = json.load(f)





WIDTH, HEIGHT = 128, 128
FLOAT_ERROR = 1e-6
assert WIDTH == HEIGHT

def normalize(strokes, frame = None):
    if frame is None:
       frame = get_frame(strokes)
    min_x, max_x, min_y, max_y = frame

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
        x_axis, y_axis, _ = stroke
        x_mod = list(map(xconv, x_axis))
        y_mod = list(map(yconv, y_axis))
        modified.append([x_mod, y_mod, None])
    return modified, frame
    
def cord_to_index(x, y):
    
    
    
    assert 0 <= x < WIDTH
    assert 0 <= y < HEIGHT
    return int(HEIGHT-y), int(x)

def line_augment(bitmap, start, end):
    xs, ys = start
    xe, ye = end
    xs, ys, xe, ye = map(int, [xs, ys, xe, ye])
    for point in bresenham.bresenham(xs, ys, xe, ye):
        xp, yp = point
        bitmap[cord_to_index(xp, yp)] = 1
    return

def stroke_to_bitmap(strokes, bitmap=None, frame=None):
    
    
    strokes, frame = normalize(strokes, frame)
    
    if bitmap is None:
        bitmap = np.zeros(shape=(HEIGHT, WIDTH), dtype=bool)
    
    for stroke in strokes:
        x_axis, y_axis, _ = stroke
        points = axis_to_points(x_axis, y_axis)
        for start, end in zip(points[:-1], points[1:]):
            
            
            if float("NaN") in start or float("NaN") in end:
                print(points)
            line_augment(bitmap, start, end)
    
    return bitmap, frame

def point_to_bitmap(point, bitmap=None, frame=None):
    
    
    
    points, frame = normalize([point], frame)
    
    if bitmap is None:
        bitmap = np.zeros(shape=(HEIGHT, WIDTH), dtype = bool)
    x, y = map(int, [points[0][0][0],points[0][1][0]])
    bitmap[cord_to_index(x,y)] = 1
 
    return bitmap, frame

def plot_bitmap(bitmap):
    bitmap = bitmap.astype(float)
    if len(bitmap.shape) == 2:
        plt.imshow(bitmap, cmap='Greys')
    elif len(bitmap.shape) == 3:
        plt.imshow(bitmap)
    plt.show()

sample_map, _ = stroke_to_bitmap(sample)
plot_bitmap(sample_map)











for randnum in [24]:
    strokes = strokes_set[randnum]
    print("", "="*50, "\n", "raw stroke")
    plot_stroke(strokes)
    print("", "="*50, "\n", "normalized/rdp stroke")
    stroke, _ = normalize(rdp_apply(strokes))
    plot_stroke(stroke)
    print("", "="*50, "\n", "normalized/rdp bitmap")
    stroke, _ = stroke_to_bitmap(strokes)
    plot_bitmap(stroke)





def stroke_to_trainset(strokes):
  random_k = np.random.randint(0, len(strokes))
  stroke_k = strokes[random_k]
  x_axis, y_axis, _ = stroke_k
  

  random_l = np.random.randint(0, len(x_axis))
  prev_stroke = [x_axis[:random_l+1], y_axis[:random_l+1], None]
  end_point = [[x_axis[random_l]], [y_axis[random_l]], [None]]

  if random_l + 1 == len(x_axis):
    next_point = [[x_axis[-1]], [y_axis[-1]], [None]]
  else:
    next_point = [[x_axis[random_l+1]], [y_axis[random_l+1]], [None]]
  
  bitmap_img, frame = stroke_to_bitmap([stroke_k])
  bitmap_prev, _ = stroke_to_bitmap([prev_stroke], frame = frame)
  bitmap_endpoint, _ = point_to_bitmap(end_point, frame = frame)
  bitmap_nextpoint, _ = normalize([next_point], frame = frame)
  bitmap_nextpoint = bitmap_nextpoint[0]
  bitmap_nextpoint = [bitmap_nextpoint[0][0], bitmap_nextpoint[1][0]]

  
  input_data = np.array([bitmap_img, bitmap_prev, bitmap_endpoint], dtype = bool) 
  input_data = np.moveaxis(input_data, 0, -1) 
  output_data = np.array(bitmap_nextpoint, dtype = np.int8)
  return input_data, output_data

sample_stroke = random.choice(strokes_set)
input_data, output_data = stroke_to_trainset(sample_stroke)
print(input_data.shape, output_data.shape)

plot_bitmap(input_data)
print("", "Visualized input data", "\n", "="*50, "\n", "Visualized output data")
print(output_data)





import tensorflow as tf
import tensorflow_addons as tfa
import keras.api._v2.keras as keras
from keras import layers

print("Num GPUs Available: ", tf.config.list_physical_devices('GPU'))



def batchnormal(name='unnamed'):
    
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
    
    
    
    
    

    custom_layers = [
        layers.Conv2D(16, (5,5), strides=(1,1), padding='same',
            kernel_initializer='he_normal', name='first-layer'), 
        batchnormal('first-batch'), relu(),
        
        IdentityBlock(filters=(16,16), name = '16c-first'),   
        IdentityBlock(filters=(16,16), name = '16c-second'),  
        IdentityBlock(filters=(16,16), name = '16c-third'),   

        ConvBlock(filters=(32,32), strides=(2,2), name = '32c-first'),  
        IdentityBlock(filters=(32,32), name = '32c-second'),  
        IdentityBlock(filters=(32,32), name = '32c-third'),   

        ConvBlock(filters=(64,64), strides=(2,2), name = '64c-first'),  
        IdentityBlock(filters=(64,64), name = '64c-second'),  
        IdentityBlock(filters=(64,64), name = '64c-third'),   

        ConvBlock(filters=(64,64), strides=(2,2), name = '64cc-first'),  
        IdentityBlock(filters=(64,64), name = '64cc-second'),  
        IdentityBlock(filters=(64,64), name = '64cc-third'),   

        ConvBlock(filters=(64,64), strides=(2,2), name = '64ccc-first'),  
        IdentityBlock(filters=(64,64), name = '64ccc-second'),  
        IdentityBlock(filters=(64,64), name = '64ccc-third'),   

        ConvBlock(filters=(128,128), strides=(2,2), name = '128c-first'),  
        IdentityBlock(filters=(128,128), name = '128c-second'),  
        IdentityBlock(filters=(128,128), name = '128c-third'),   

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


CIFARDNNModel = Resnet()
CIFARDNNModel.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss=keras.losses.MeanSquaredError(),
    metrics=['mean_absolute_error', 'mean_absolute_percentage_error']
)

print(CIFARDNNModel.summary())






print("Total dataset size: ", len(strokes_set))
train_size = len(strokes_set)
train_strokeset = (strokes_set)

input_dataset, output_dataset = np.zeros((train_size, HEIGHT, WIDTH, 3)), np.zeros((train_size, 2))

for i in tqdm(range(len(train_strokeset))):
    strokes = train_strokeset[i]
    input_single, output_single = stroke_to_trainset(strokes)
    input_dataset[i], output_dataset[i] = input_single, output_single.flatten()

dataset = tf.data.Dataset.from_tensor_slices(
    (input_dataset, output_dataset)
).shuffle(100).batch(4)

print(dataset)

val_size = 100
val_strokeset = random.sample(strokes_set, val_size)

input_dataset, output_dataset = np.zeros((val_size, HEIGHT, WIDTH, 3)), np.zeros((val_size, 2))

for i in tqdm(range(len(val_strokeset))):
    strokes = val_strokeset[i]
    input_single, output_single = stroke_to_trainset(strokes)
    input_dataset[i], output_dataset[i] = input_single, output_single.flatten()

val_dataset = tf.data.Dataset.from_tensor_slices(
    (input_dataset, output_dataset)
).batch(val_size)
































print(len(strokes_set))






def scheduler(epoch, _lr):
    if epoch < 32 * 3:
        return 1e-4
    elif epoch < 48 * 3:
        return 1e-5
    else:
        return 1e-6

lr_modify = keras.callbacks.LearningRateScheduler(scheduler)


history = CIFARDNNModel.fit(
    dataset,
    epochs=64 * 3, batch_size=128,
    
    
)







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
    plt.show()



plot_history(history)







