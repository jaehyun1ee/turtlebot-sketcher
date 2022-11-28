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
CIFARDNNModel.load_weights('./output/checkpoint.mdl').expect_partial()

# Provide ramdom img & starting point to draw 
# Output: bitmap img to draw, list of start position [x,y], list of end position [x,y]
def get_random_img(path):
    
    CALCULATE_MODE = prepare_data()
    
    if CALCULATE_MODE:
        print("Processing from /data...")
        with open(path) as f:
            print("--- Loading file: {}".format(f))
            random_drawings = json.load(f)
        random_stroke = random.choice(random.choice(random_drawings))
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

# Plot any img represented by 128x128 bitmap
def plot_img(bitmap_img):
    plot_bitmap(bitmap_img, "img_display")

# Convert img to strokes 
# input: bitmap img, start point list [x,y]
# output: list of points to draw [[x1, y1], [x2, y2], ..., [xn, yn]]
def img_to_strokes(bitmap_img, start_point, end_point):
    goal_bitmap = bitmap_img
    cur_bitmap = point_to_bitmap(start_point)
    point_bitmap = cur_bitmap
    
    cur_point = start_point
    x_values = [cur_point[0]]
    y_values = [cur_point[1]]
    #while True:
    for i in range (15):
        dist = (cur_point[0]-end_point[0])**2 + (cur_point[1]-end_point[1])**2
        if dist < 10: break
        
        next_point = commander_get_nextpoint(goal_bitmap, cur_bitmap, point_bitmap)[0]
        cur_bitmap = stroke_to_bitmap([[x_values, y_values]])
        point_bitmap = point_to_bitmap(next_point)
        x_values.append(next_point[0])
        y_values.append(next_point[1])
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
