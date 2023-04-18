# %%
import numpy as np
from data_processing import get_data

# %%
dataset_name = 'york'
normal_data_file = f"./data/{dataset_name}/{dataset_name}_gt_data_old.csv"
normal_data = np.genfromtxt(normal_data_file, delimiter=',')

# %%
normal_data = np.repeat(normal_data, 100, axis=0)

# %%
# Remove id
id = normal_data[:, 0]
id = np.reshape(id, [len(id), -1])
normal_data = normal_data[:, 1:]
# Remove category
categories = normal_data[:, 0]
categories = np.reshape(categories, [len(categories), -1])
normal_data = normal_data[:, 1:]

# %%
distr = np.random.normal(0, 5, normal_data.shape)

# %%
normal_data = np.add(normal_data, distr)

# %%
normal_data = np.concatenate([categories, normal_data], axis=1)
normal_data = np.concatenate([id, normal_data], axis=1)

# %%
np.savetxt(f"./data/{dataset_name}/{dataset_name}_gt_data.csv", normal_data, delimiter=',')
