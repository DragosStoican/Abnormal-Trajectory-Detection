import numpy as np
import matplotlib.pyplot as plt


# Plot the data shape in a barplot
def plot(normal_data, abnormal_data, real_abnormal_data, real_abnormal_data_2):
    weights = {
        "normal": np.array([normal_data.shape[0], 0]),
        "abnormal": np.array([0, abnormal_data.shape[0]]),
        "real_abnormal": np.array([0, real_abnormal_data.shape[0]]),
        "real_abnormal_2": np.array([0, real_abnormal_data_2.shape[0]]),
    }

    abnormal_size = abnormal_data.shape[0] + real_abnormal_data.shape[0] + real_abnormal_data_2.shape[0]

    fig, ax = plt.subplots()
    bottom = np.zeros(2)

    for traj, weight in weights.items():
        p = ax.bar([f'Normal = {normal_data.shape[0]}', f'Abnormal = {abnormal_size}'], weight, label=traj,
                   bottom=bottom)
        bottom += weight

    plt.legend()
    plt.show()


if __name__ == '__main__':
    DATASET_NAME = "stmarc"
    normal_data, abnormal_data, real_abnormal_data, real_abnormal_data_2 = get_data(DATASET_NAME)
