# %%
import matplotlib.pyplot as plt
from data_processing import get_data
import numpy as np


# %%
def show_trajectories(image, trajectories, color):
    fig = plt.figure()
    ax = plt.axes()

    background = plt.imread(image, format='png')
    ax.imshow(background)

    # Plot trajectory data
    for i, trajectory in enumerate(trajectories):
        print(f"Plotting trajectory {i + 1}/{len(trajectories)}")
        trajectory_data = trajectory[1:]  # Remove category
        category = int(trajectory[0])
        x_positions = trajectory_data[0::2]
        y_positions = trajectory_data[1::2]

        ax.plot(x_positions, y_positions, '-', linewidth=1, color=color[category])
        ax.arrow(x_positions[-2], y_positions[-2],
                 x_positions[-1] - x_positions[-2], y_positions[-1] - y_positions[-2],
                 head_width=20, head_length=10, fc=color[category], ec=color[category])

    plt.show()

    print("Done")


# %%
dataset = 'york'
image = 'data/images/' + dataset + '.png'
normal_data, abnormal_data, real_abnormal_data, real_abnormal_data_2 = get_data(dataset, scale=False)

# %%
# for i in range(len(normal_data)):
#     for j in range(2, len(normal_data[i][1:])+1, 2):
#         if j % 2 == 0:
#             normal_data[i, j] = (450 - normal_data[i, j]) % 450

# %%
show_trajectories(image, normal_data, ['green', 'blue', 'yellow'])
# show_trajectories(image, abnormal_data, ['green', 'blue', 'yellow'])
# show_trajectories(image, real_abnormal_data, ['green', 'blue', 'yellow'])
# show_trajectories(image, real_abnormal_data_2, ['green', 'blue', 'yellow'])
