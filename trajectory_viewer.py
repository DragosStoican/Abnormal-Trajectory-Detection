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
        trajectory_data = trajectory[1:]  # Remove category and label
        category = int(trajectory[0])
        x_positions = trajectory_data[0::4]
        y_positions = trajectory_data[1::4]

        ax.plot(x_positions, y_positions, '-', linewidth=1, color=color[category])
        ax.arrow(x_positions[-2], y_positions[-2],
                 x_positions[-1] - x_positions[-2], y_positions[-1] - y_positions[-2],
                 head_width=20, head_length=10, fc=color[category], ec=color[category])

    plt.show()

    print("Done")


# %%
dataset = 'sherbrooke'
image = 'data/images/' + dataset + '.png'
normal_data, abnormal_data, real_abnormal_data, real_abnormal_data_2 = get_data(dataset, scale=False)

# %%
# show_trajectories(image, normal_data, ['green', 'blue', 'yellow'])
show_trajectories(image, abnormal_data, ['green', 'blue', 'yellow'])
show_trajectories(image, real_abnormal_data, ['green', 'blue', 'yellow'])
show_trajectories(image, real_abnormal_data_2, ['green', 'blue', 'yellow'])
