from data_processing import get_data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

print(tf.config.list_physical_devices('GPU'))

normal_data, abnormal_data, real_abnormal_data, real_abnormal_data_2 = get_data("stmarc")

data = np.concatenate([normal_data, abnormal_data, real_abnormal_data, real_abnormal_data_2])

labels = np.concatenate([
    np.ones(normal_data.shape[0]),
    np.zeros(abnormal_data.shape[0]),
    np.zeros(real_abnormal_data.shape[0]),
    np.zeros(real_abnormal_data_2.shape[0])
], 0)
labels = np.reshape(labels, [-1, 1])

neg, pos = tf.math.bincount(labels)
total = neg + pos
print('Examples:\n    Total: {}\n    Normal: {} ({:.2f}% of total)\n'.format(
    total, pos, 100 * pos / total))

epss = np.arange(0.1, 1.1, 0.1)
vals = {
    "Normal acc": [],
    "Abnormal acc": [],
    "Mean acc": []
}
for eps in epss:
    db = DBSCAN(eps=eps).fit(data)
    labels_cluster = db.labels_
    labels_cluster = np.reshape(labels_cluster, [-1, 1])

    labels_cluster[labels_cluster != -1] = 1
    labels_cluster[labels_cluster == -1] = 0

    # Let's see the distribution of normal and abnormal data

    diff = labels - labels_cluster
    correct_normal = len(diff[(diff == 0) & (labels == 1)]) * 100 / pos
    correct_abnormal = len(diff[(diff == 0) & (labels == 0)]) * 100 / neg
    mean = (correct_normal + correct_abnormal) / 2
    print(f"Normal accuracy: {correct_normal:.2f}")
    print(f"Abnormal accuracy: {correct_abnormal:.2f}")
    print(f"Mean accuracy: {mean:.2f}")

    vals["Normal acc"].append(correct_normal)
    vals["Abnormal acc"].append(correct_abnormal)
    vals["Mean acc"].append(mean)

# %%
plt.rcParams.update({'font.size': 15})
plt.plot(epss, vals["Normal acc"], color='green', label='normal')
plt.plot(epss, vals["Abnormal acc"], color='red', label='abnormal')
plt.plot(epss, vals["Mean acc"], color='blue', label='mean')
plt.xlabel('eps')
plt.ylabel('accuracy')
plt.legend()
plt.show()

# %%
print(np.argmax(vals["Mean acc"]))
    # correct = len(diff[diff == 0])
    # false_neg = len(diff[diff == 1])
    # false_pos = len(diff[diff == -1])
    #
    # plt.rcParams.update({'font.size': 18})
    # # plt.xticks([0, 1])
    # plt.bar(["Correct", "False Neg", "False Pos"], [correct, false_neg, false_pos])
    # plt.show()