import utils.data_utils as datautils
import utils.abnormal_data_generation as adg
import numpy as np
import os

#%%
# Define dataset locations
RAW_DATASET_PATH = './data/sherbrooke/raw/sherbrooke'
RAW_DATASET = RAW_DATASET_PATH + '_gt.sqlite'
RAW_PEDS = RAW_DATASET_PATH + '_gt_pedestrians.sqlite'
RAW_CARS = RAW_DATASET_PATH + '_gt_cars.sqlite'
RAW_FILE_NAMES = [RAW_PEDS, RAW_CARS]
RAW_FRAMES = RAW_DATASET_PATH + '_frames/'
VIDEO_DATA_FPS = 30

# Generate augmented normal data
dataset_file = datautils.extract_augment_and_export_data(raw_input_file_all=RAW_DATASET,
                                                         input_raw_image_frame_path=RAW_FRAMES,
                                                         raw_input_file_names=RAW_FILE_NAMES,
                                                         video_data_fps=VIDEO_DATA_FPS,
                                                         generate_graph=False,
                                                         show_graph=False)
#%%
# Create datasets
normal_data = np.genfromtxt(dataset_file, delimiter=',')

# Remove the first column, which is the object_id
normal_data = normal_data[1:, 1:]
# dataset.shape
# (20605, 125)
# The format of the dataset is:
# [label, x1, y1, vx1, vy1, x2, y2, vx2, vy2, ..., x31, y31, vx31 , vy31 ]

# Generate abnormal data
abnormal_data = adg.generate_abnormal_data(n_objects=20,
                                           raw_input_file_all=RAW_DATASET)
abnormal_data = abnormal_data[:,1:]
# abnormal_data.shape
# (186, 125)
#%%

