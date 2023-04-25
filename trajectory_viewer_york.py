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

        linewidth = 1
        ax.plot(x_positions, y_positions, '-', linewidth=linewidth, color=color[category])
        ax.arrow(x_positions[-2], y_positions[-2],
                 x_positions[-1] - x_positions[-2], y_positions[-1] - y_positions[-2],
                 head_width=20*linewidth, head_length=10*linewidth, fc=color[category], ec=color[category])

    plt.show()

    print("Done")


# %%
dataset = 'york'
image = 'data/images/' + dataset + '.png'
normal_data_file = f"./data/{dataset}/{dataset}_gt_data_original.csv"
normal_data = np.genfromtxt(normal_data_file, delimiter=',')
# Remove the first column, which is the object_id
normal_data = normal_data[:, 1:]
print(f"Normal data shape: {normal_data.shape}")

# Create the generated abnormal dataset
abnormal_data_file = f"./data/{dataset}/{dataset}_gt_abnormal_original.csv"
abnormal_data = np.genfromtxt(abnormal_data_file, delimiter=',')
# Remove the first column, which is the object_id
abnormal_data = abnormal_data[:, 1:]
print(f"Abnormal data shape: {abnormal_data.shape}")

# %%
show_trajectories(image, normal_data, ['green', 'blue', 'yellow'])
# show_trajectories(image, abnormal_data, ['green', 'blue', 'yellow'])
# show_trajectories(image, real_abnormal_data, ['green', 'blue', 'yellow'])
# show_trajectories(image, real_abnormal_data_2, ['green', 'blue', 'yellow'])
