"""
This script generates abnormal trajectory data for the specific dataset.
"""

import numpy as np
import os
from . import data_utils as datautils
# import dataset_defines as dd
# import trajectory_viewer as tv
# import interactive_abnormality_generator as iag


def generate_abnormal_data(n_objects,
                           raw_input_file_all,
                           generate_graph=False, show_graph=False):
    """
    Generates some abnormal trajectories like pedestrians crossing the road.
    Note: Refer to the image showed by show_image.py.
    It is worth mentioning that the y axis is inverted, which means that,
    when we say object going up, we expect y value to decrease and vise versa.
    :return: numpy array of abnormal trajectories with new objects
    """
    # Print a starting message
    print("=========================================================")
    print("Generating abnormal trajectories for sherbrooke dataset &")
    print("Exporting them to CSV files.")

    # Trajectory image viewer
    # trajectory_image = tv.ImageViewer(input_raw_image_frame_path)

    # Generate abnormal trajectory data of a pedestrian going -------->
    x = range(240, 790)
    x = np.array(x).reshape(-1, 1)
    y = np.full((x.shape[0], 1), 400)
    v_x = np.full((x.shape[0], 1), 10)
    v_y = np.full((x.shape[0], 1), 0)
    object_id = n_objects
    object_label = 0
    array_dataset = datautils._rearrange_data(object_id, object_label, x, y, v_x, v_y)
    # generate_graph=generate_graph,
    # trajectory_image=trajectory_image)

    x = range(230, 600)
    x = np.array(x).reshape(-1, 1)
    y = np.full((x.shape[0], 1), 250)
    v_x = np.full((x.shape[0], 1), 10)
    v_y = np.full((x.shape[0], 1), 0)
    object_id += 1
    object_label = 0
    array_dataset_2 = datautils._rearrange_data(object_id, object_label, x, y, v_x, v_y)
    # generate_graph=generate_graph,
    # trajectory_image=trajectory_image)
    array_dataset = np.concatenate((array_dataset, array_dataset_2), axis=0)

    # Function that generates all the values of a line
    def _gen_line_values(x_1, x_2, y_1, y_2, v_x_val):
        """
        y = m * x + b
        :param x_1:
        :param x_2:
        :param y_1:
        :param y_2:
        :param v_x_val:
        :return: x, y, v_x, v_y
        """
        m = (y_1 - y_2) / (x_1 - x_2)
        b = (x_1 * y_2 - x_2 * y_1) / (x_1 - x_2)
        x = range(int(x_1), int(x_2) + 1) if x_1 < x_2 else range(int(x_1), int(x_2) - 1, -1)
        x = np.array(x).reshape(-1, 1)
        y = m * x + b
        v_y_val = m * v_x_val
        v_x = np.full((x.shape[0], 1), v_x_val)
        v_y = np.full((x.shape[0], 1), v_y_val)
        return x, y, v_x, v_y

    # Generate abnormal trajectory data of a pedestrian going down
    # y = m * x + b
    x_1 = 310.0
    x_2 = 500.0
    y_1 = 240.0
    y_2 = 540.0
    x, y, v_x, v_y = _gen_line_values(x_1, x_2, y_1, y_2, v_x_val=5)
    object_id += 1
    object_label = 0
    array_dataset_2 = datautils._rearrange_data(object_id, object_label, x, y, v_x, v_y)
    # generate_graph=generate_graph,
    # trajectory_image=trajectory_image)
    array_dataset = np.concatenate((array_dataset, array_dataset_2), axis=0)

    # Generate abnormal trajectory data of a pedestrian going up
    # y = m * x + b
    x_1 = 760.0
    x_2 = 420.0
    y_1 = 420.0
    y_2 = 230.0
    x, y, v_x, v_y = _gen_line_values(x_1, x_2, y_1, y_2, v_x_val=-7)
    object_id += 1
    object_label = 0
    array_dataset_2 = datautils._rearrange_data(object_id, object_label, x, y, v_x, v_y)
    # generate_graph=generate_graph,
    # trajectory_image=trajectory_image)
    array_dataset = np.concatenate((array_dataset, array_dataset_2), axis=0)

    # Generate abnormal trajectory data of a car going the wrong direction
    # Left side of the road
    x_1 = 500.0
    x_2 = 310.0
    y_1 = 500.0
    y_2 = 230.0
    x, y, v_x, v_y = _gen_line_values(x_1, x_2, y_1, y_2, v_x_val=-50)
    object_id += 1
    object_label = 1
    array_dataset_2 = datautils._rearrange_data(object_id, object_label, x, y, v_x, v_y)
    # generate_graph=generate_graph,
    # trajectory_image=trajectory_image)
    array_dataset = np.concatenate((array_dataset, array_dataset_2), axis=0)

    # Right side of the road
    x_1 = 420.0
    x_2 = 760.0
    y_1 = 230.0
    y_2 = 420.0
    x, y, v_x, v_y = _gen_line_values(x_1, x_2, y_1, y_2, v_x_val=50)
    object_id += 1
    object_label = 1
    array_dataset_2 = datautils._rearrange_data(object_id, object_label, x, y, v_x, v_y)
    # generate_graph=generate_graph,
    # trajectory_image=trajectory_image)
    array_dataset = np.concatenate((array_dataset, array_dataset_2), axis=0)

    # Export it to CSV file
    dataset_name = raw_input_file_all[raw_input_file_all.rfind('/') + 1:raw_input_file_all.rfind('_')]
    dataset_file_name = 'data/' + dataset_name + '/augmented/' + dataset_name + '_abnormal.csv'

    directory = os.path.join(datautils.dir_name, dataset_file_name[:dataset_file_name.rfind('/')])
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(os.path.join(datautils.dir_name, dataset_file_name), 'wb') as datafile:
        np.savetxt(datafile, array_dataset, fmt="%.2f", delimiter=",")

    # Save and Show the trajectory image
    # if generate_graph:
    #     trajectory_image_name = directory + '/' + \
    #                             raw_input_file_all[raw_input_file_all.rfind('/')+1:
    #                             raw_input_file_all.rfind('.')] + \
    #                             '_abnormal_trajectories.pdf'
    #     trajectory_image.save_image(os.path.join(datautils.dir_name, trajectory_image_name))
    # if generate_graph and show_graph:
    #     trajectory_image.show_image()

    # Print finishing message
    print("                                               ---> Done!")
    print("=========================================================")

    return array_dataset


# def generate_abnormal_data_from_raw_data(input_raw_image_frame_path,
#                                          overwrite_data=True,
#                                          generate_graph=False, show_graph=False):
#     """
#     Generate abnormal trajectory data from raw trajectories by translating x and y positions.
#     :param generate_graph:
#     :param show_graph:
#     :return:
#     """
#     dataset_file_name = 'data/' + \
#                         raw_input_file_all[raw_input_file_all.rfind('/') + 1:raw_input_file_all.rfind('.')] \
#                         + '_real_abnormal.csv'
#
#     if overwrite_data or not os.path.isfile(dataset_file_name):
#         iag.extract_and_drag_trajectories(raw_input_file_all=raw_input_file_all,
#                                           input_raw_image_frame_path=input_raw_image_frame_path,
#                                           raw_input_file_names=dd.raw_input_file_names,
#                                           video_data_fps=dd.video_data_fps,
#                                           generate_graph=generate_graph, show_graph=show_graph)
#
#     array_dataset = np.genfromtxt(os.path.join(datautils.dir_name, dataset_file_name), delimiter=',')
#
#     return array_dataset
