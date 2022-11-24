import data.data_utils as datautils

# Define dataset locations
RAW_DATASET_PATH = '/data/sherbrooke/raw/sherbrooke'
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
