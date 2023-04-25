import numpy as np
from data_processing import get_data

# %%
arr = np.genfromtxt('data/york/traj.csv', delimiter=',')

# print(arr)

# arr = np.flip(arr, axis=0)




# y = arr[::2]
# x = arr[1::2]



# arr = np.dstack((x, y))
# arr = arr.flatten()

np.savetxt('test.csv', arr)

# %%
# for j in range(0, len(arr), 2):
#     arr[j] = (450 - arr[j]) % 450
# %%
# normal_data, abnormal_data, real_abnormal_data, real_abnormal_data_2 = get_data('york', scale=False)
#
# for i in range(len(normal_data)):
#     for j in range(2, len(normal_data[i][1:])+1, 2):
#         if j % 2 == 0:
#             normal_data[i, j] = (450 - normal_data[i, j]) % 450
#
# np.savetxt('york_gt_data.csv', normal_data, delimiter=',')

# %%
arr = np.genfromtxt('test.csv', delimiter=',')

arr = np.flip(arr, axis=1)

for i, a in enumerate(arr):
    y = a[::2]
    x = a[1::2]

    tmp = np.dstack((x, y))
    tmp = tmp.flatten()
    arr[i] = tmp

np.savetxt('test2.csv', arr, delimiter=',')


