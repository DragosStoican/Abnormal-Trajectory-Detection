# %%
import matplotlib.pyplot as plt
from data_processing import get_data
import numpy as np


# %%
def show_trajectories(image, normal_trajectories, abnormal_trajectories):
    fig = plt.figure()
    ax = plt.axes()

    background = plt.imread(image, format='png')
    ax.imshow(background)

    # Plot trajectory data
    _show_trajectories(ax, normal_trajectories, 'green')

    plt.show()

    print("Done")


def _show_trajectories(ax, trajectories, color):
    last_x_positions = np.zeros(31)
    last_y_positions = np.zeros(31)
    for i, trajectory in enumerate(trajectories):

        trajectory_data = trajectory[1:-1]  # Remove category and label
        x_positions = trajectory_data[0::4]
        y_positions = trajectory_data[1::4]

        diff = abs(x_positions[0] - last_x_positions[0]) + abs(y_positions[0] - last_y_positions[0])
        if diff >= 0:
            print(f"Plotting trajectory {i + 1}/{len(trajectories)}")
            ax.plot(x_positions, y_positions, '-', linewidth=0.3, color=color)
            ax.arrow(x_positions[-2], y_positions[-2],
                     x_positions[-1] - x_positions[-2], y_positions[-1] - y_positions[-2],
                     head_width=5, head_length=2.5, fc=color, ec=color)

        last_x_positions = x_positions
        last_y_positions = y_positions


# %%
dataset = 'sherbrooke'
image = 'data/images/' + dataset + '.png'
normal_data, abnormal_data, real_abnormal_data, real_abnormal_data_2 = get_data(dataset, scale=False)

abnormal_trajectories = np.concatenate([abnormal_data, real_abnormal_data, real_abnormal_data_2])

# labels = np.concatenate([
#     np.ones(normal_data.shape[0]),
#     np.zeros(abnormal_data.shape[0]),
#     np.zeros(real_abnormal_data.shape[0]),
#     np.zeros(real_abnormal_data_2.shape[0])
# ], 0)
# labels = np.reshape(labels, [-1, 1])

# trajectories = np.concatenate([trajectories, labels], 1)

# %%
show_trajectories(image, normal_data, abnormal_trajectories)
