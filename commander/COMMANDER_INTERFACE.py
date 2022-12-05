from commander.COMMANDER import * 
import random

print("load weights...")
CIFARDNNModel = Resnet()
"""
CIFARDNNModel.compile(
    optimizer=tfa.optimizers.AdamW(
        learning_rate=1e-4,
        weight_decay=1e-4
    ),
    loss=keras.losses.MeanSquaredError(),
    metrics=['mean_absolute_error', 'mean_absolute_percentage_error']
)
"""
print("Model prepared!")
CIFARDNNModel.load_weights('./model/commander/checkpoint.mdl').expect_partial()

# Provide ramdom img & starting point to draw 
# Output: bitmap img to draw, list of start position [x,y], list of end position [x,y]
def get_random_img(path):
    
    CALCULATE_MODE = prepare_data()
    
    if CALCULATE_MODE:
        print("Processing from /data...")
        with open(path) as f:
            print("--- Loading file: {}".format(f))
            random_drawings = json.load(f)
        random_stroke = random.choice(random_drawings)
        bitmap_img = stroke_to_bitmap([random_stroke])
        start_point = np.array([random_stroke[0][0], random_stroke[1][0]])
        end_point = np.array([random_stroke[0][-1], random_stroke[1][-1]])

        print("Saving processed results into .npz format...")
        np.save('../output/bitmap_img', bitmap_img)
        np.save('../output/start_point', start_point)
        np.save('../output/end_point', end_point)
    else:
        print("Loading processed cache...")
        bitmap_img = np.load('../output/bitmap_img.npy')
        start_point = np.load('../output/start_point.npy')
        end_point = np.load('../output/end_point.npy')
    
    return bitmap_img, start_point, end_point

# Get strokes from the path
# Output: list of strokes / stroke in format [[x1 .. xn][y1 .. yn]]
def get_img_strokes(path):
    print("Processing ...")
    stroke_list = np.load(path, allow_pickle=True)
    return stroke_list

# Convert a single stroke to bitmap img
def stroke2img(stroke):
    bitmap_img = stroke_to_bitmap([stroke])
    return bitmap_img

def stroke2points(stroke):
    return axis_to_points(stroke[0], stroke[1])

# Plot any img represented by 128x128 bitmap
def plot_img(bitmap_img, name):
    plot_bitmap(bitmap_img, path)

# Convert img to strokes 
# input: bitmap img, start point list [x,y]
# output: list of points to draw [[x1, y1], [x2, y2], ..., [xn, yn]]
def img2points(goal_bitmap, start_point, end_point):    
    
    cur_bitmap = point_to_bitmap(start_point)
    point_bitmap = cur_bitmap
        
    x_values = []
    y_values = []
    x_values.append(start_point[0])
    y_values.append(start_point[1])

    max_iter = 60
    for i in range(max_iter):
        next_point = commander_get_nextpoint(goal_bitmap, cur_bitmap, point_bitmap)[0]
        
        x_values.append(next_point[0])
        y_values.append(next_point[1])
        
        cur_bitmap = stroke_to_bitmap([[x_values, y_values]])    
        point_bitmap = point_to_bitmap(next_point)
            
        dist = (end_point[0]-next_point[0])**2 + (end_point[1]-next_point[1])**2
        if i > 5 and dist < 45:
            break
    return axis_to_points(x_values, y_values)

def commander_get_nextpoint(goal_bitmap, current_bitmap, cur_point):
    def input_gen():
        yield input_data[0]
    
    input_data = []
    data = np.array([goal_bitmap, current_bitmap, cur_point], dtype = bool)
    data = np.moveaxis(data, 0, -1) # (height, width, channel) order -> suitable for Tensorflow
    input_data.append(data)
    
    input_dataset = tf.data.Dataset.from_generator(
        input_gen, output_types=(bool), output_shapes=([HEIGHT, WIDTH, 3])
    ).batch(1)
    
    output = CIFARDNNModel.predict(input_dataset)
    return output

def visualize_commander(points_list, path):
    stroke_list = list(map(points_to_axis, points_list))
    bitmap_img = stroke_to_bitmap(stroke_list)
    plot_bitmap(bitmap_img, path)
