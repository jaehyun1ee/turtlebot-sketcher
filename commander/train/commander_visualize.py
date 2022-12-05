import tensorflow as tf
import tensorflow_addons as tfa
import keras.api._v2.keras as keras
from keras import layers
from COMMANDER import * 
from COMMANDER_INTERFACE import *

def load_testset(test_num):
    def test_gen():
        for idx in range(test_size):
            yield test_input[idx], test_output[idx]

    CALCULATE_MODE = prepare_data()
    input_dataset, output_dataset = [], []    
    print("\nLoading testset...")
    if CALCULATE_MODE:
        print("Processing from /data...")
        path_single = drawings_path[test_num]
        with open(path_single) as f:
            print("--- Loading file: {}".format(f))
            drawings = json.load(f)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for input_data, output_data in tqdm(executor.map(stroke_to_trainset, drawings), total=len(drawings)):
                input_dataset.append(input_data)
                output_dataset.append(output_data.flatten())

        print("Saving processed results into .npz format...")
        np.save('../output/test_input_dataset', input_dataset)
        np.save('../output/test_output_dataset', output_dataset)
    else:
        print("Loading processed cache...")
        input_dataset = np.load('../output/test_input_dataset.npy')
        output_dataset = np.load('../output/test_output_dataset.npy')
    
    test_input = input_dataset[:test_size]
    test_output = output_dataset[:test_size]
    
    test_dataset = tf.data.Dataset.from_generator(
        test_gen, output_types=(bool, np.int8), output_shapes=([HEIGHT, WIDTH, 3], [2])
    ).batch(32)


    print("****** TRAINING INFORMATIONS ******")
    print("Test size:", test_size)
    print("***********************************")
    
    return test_dataset


"""
print("load weights...")
CIFARDNNModel = Resnet()
CIFARDNNModel.compile(
    optimizer=tfa.optimizers.AdamW(
        learning_rate=1e-4,
        weight_decay=1e-4
    ),
    loss=keras.losses.MeanSquaredError(),
    metrics=['mean_absolute_error', 'mean_absolute_percentage_error']
)
print("Model prepared!")
CIFARDNNModel.load_weights('../output/checkpoint.mdl').expect_partial()
"""


def test ():
    print("Test model")
    test_dataset = load_testset(test_num)
    test_loss = CIFARDNNModel.evaluate(test_dataset)
    print(test_loss)
    

def visualize(draw_num = 0, only_img = False):
    print("visualize")

    #print(drawings_path)
    test_path = drawings_path[test_num]
    with open(test_path) as f:
        print("--- Loading file: {}".format(f))
        drawings = json.load(f)
        
    # visualize the original img
    output_strokes = []
    print("stroke numbers: ", len(drawings[draw_num]))
    plot_bitmap(stroke_to_bitmap(drawings[draw_num]), "original_img_{}_{}".format(test_num, draw_num))
    stroke_inputs = list(map(one_axis_to_points, drawings[draw_num]))
    np.save("../output/whole_drawing_original_values_{}th_test".format(test_num), stroke_inputs)

    if only_img:
        return 

    for strokes_i in range (len(drawings[draw_num])):
        single_drawing = drawings[draw_num][strokes_i]
      #  np.save("../output/original_values_{}th test_{}th stroke".format(test_num, strokes_i), axis_to_points(single_drawing[0], single_drawing[1]))
        original_bitmap = stroke_to_bitmap([single_drawing])
        plot_bitmap(original_bitmap, "original_img_{}th test_{}th stroke".format(test_num, strokes_i))
            
        # visualize the model performance
        start_point = [single_drawing[0][0], single_drawing[1][0]]
        end_point = [single_drawing[0][-1], single_drawing[1][-1]]
        goal_bitmap = original_bitmap
        cur_bitmap = point_to_bitmap(start_point)
        point_bitmap = cur_bitmap
        

        x_values = []
        y_values = []
        x_values.append(start_point[0])
        y_values.append(start_point[1])

        max_iter = 60
        for i in range(max_iter):
            next_point = commander_get_nextpoint(goal_bitmap, cur_bitmap, point_bitmap)[0]
            #print(x_values, y_values)
            x_values.append(next_point[0])
            y_values.append(next_point[1])
            #next_stroke = points_to_axis([cur_point, next_point])
            #stroke_bitmap = stroke_to_bitmap([next_stroke])
            new_bitmap = stroke_to_bitmap([[x_values, y_values]])
            plot_bitmap(new_bitmap, "stroke_process/stroke_output in stroke" + str(i))

            cur_bitmap = new_bitmap
            point_bitmap = point_to_bitmap(next_point)
            
        #  print(x_values, y_values)
            dist = (end_point[0]-next_point[0])**2 + (end_point[1]-next_point[1])**2
            if i > 5 and dist < 45:
                break
        # Print the point
        #bitmap = point_to_bitmap(next_point[0])
        #plot_bitmap(bitmap, "model_output")
        output_strokes.append([x_values, y_values])
        drawed_bitmap = stroke_to_bitmap(output_strokes)
        plot_bitmap(drawed_bitmap, "drawed_bitmap_{}th test_{}th drawing_{}th stroke".format(test_num, draw_num, strokes_i))
        #np.save('../output/stroker_input', axis_to_points(x_values, y_values))
    stroke_inputs = list(map(one_axis_to_points, output_strokes))
    np.save("../output/whole_drawing_commander_values_{}th_test".format(test_num), stroke_inputs)


test_size = 10000
test_num = 15
# test()