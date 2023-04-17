import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler


def get_data(dataset_name, scale=True):
    # Create normal trajectories dataset
    normal_data_file = f"./data/{dataset_name}/{dataset_name}_gt_data.csv"
    normal_data = np.genfromtxt(normal_data_file, delimiter=',')
    # Remove the first column, which is the object_id
    normal_data = normal_data[:, 1:]
    print(f"Normal data shape: {normal_data.shape}")

    # Create the generated abnormal dataset
    abnormal_data_file = f"./data/{dataset_name}/{dataset_name}_gt_abnormal.csv"
    abnormal_data = np.genfromtxt(abnormal_data_file, delimiter=',')
    # Remove the first column, which is the object_id
    abnormal_data = abnormal_data[:, 1:]
    print(f"Abnormal data shape: {abnormal_data.shape}")

    # Create the real abnormal datasets
    real_abnormal_file = f"./data/{dataset_name}/{dataset_name}_gt_real_abnormal.csv"
    real_abnormal_file_2 = f"./data/{dataset_name}/{dataset_name}_gt_real_abnormal_2.csv"
    real_abnormal_data = np.genfromtxt(real_abnormal_file, delimiter=',')
    real_abnormal_data_2 = np.genfromtxt(real_abnormal_file_2, delimiter=',')
    # Remove the first column, which is the object_id
    real_abnormal_data = real_abnormal_data[:, 1:]
    real_abnormal_data_2 = real_abnormal_data_2[:, 1:]
    print(f"Real abnormal data shape: {real_abnormal_data.shape}")
    print(f"Real abnormal data 2 shape: {real_abnormal_data_2.shape}")

    # Scale the data
    if not scale:
        return normal_data, abnormal_data, real_abnormal_data, real_abnormal_data_2

    scaler = MinMaxScaler()
    normal_data_scaled = scaler.fit_transform(normal_data)
    # normal_data_scaled = torch.from_numpy(normal_data_scaled).to(dtype=torch.float32)

    abnormal_data_scaled = scaler.fit_transform(abnormal_data)
    # abnormal_data_scaled = torch.from_numpy(abnormal_data_scaled).to(dtype=torch.float32)

    abnormal_data_real_scaled = scaler.fit_transform(real_abnormal_data)
    # abnormal_data_real_scaled = torch.from_numpy(abnormal_data_real_scaled).to(dtype=torch.float32)

    abnormal_data_real_scaled_2 = scaler.fit_transform(real_abnormal_data_2)
    # abnormal_data_real_scaled_2 = torch.from_numpy(abnormal_data_real_scaled_2).to(dtype=torch.float32)

    return normal_data_scaled, abnormal_data_scaled, abnormal_data_real_scaled, abnormal_data_real_scaled_2


if __name__ == '__main__':
    normal_data, abnormal_data, real_abnormal_data, real_abnormal_data_2 = get_data("sherbrooke")
