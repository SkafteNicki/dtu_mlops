import torch
import torch.utils.data as data_utils


import numpy as np
from pathlib import Path
import os
import matplotlib.pyplot as plt


ROOT_PATH = Path(__file__).parent.parent.parent.parent
DATA_PATH = os.path.join(ROOT_PATH, "data/corruptmnist")


def mnist():
    train_image_list = []
    train_label_list = []
    for i in range(5):
        img, labels = get_data(f"train_{i}.npz")
        train_image_list.append(img)
        train_label_list.append(labels)
    train_images = torch.as_tensor(train_image_list).view(5*5000, 28, 28)
    train_labels = torch.as_tensor(train_label_list).view(5*5000, -1).flatten()
    train = data_utils.TensorDataset(train_images, train_labels)

    test_image_list, test_label_list = get_data("test.npz")
    test_images = torch.as_tensor(test_image_list)
    test_labels = torch.as_tensor(test_label_list).flatten()
    test = data_utils.TensorDataset(test_images, test_labels)
    return train, test


def get_data(file_name):
    data_file = os.path.join(DATA_PATH, file_name)
    data = np.load(data_file)
    return [data["images"], data["labels"]]


def plot_img(x):
    i = 1000
    plt.figure(figsize=(10, 10))
    plt.subplot(221), plt.imshow(x[i], cmap='gray')
    plt.subplot(222), plt.imshow(x[i + 25], cmap='gray')
    plt.subplot(223), plt.imshow(x[i + 50], cmap='gray')
    plt.subplot(224), plt.imshow(x[i + 75], cmap='gray')
    plt.show()


