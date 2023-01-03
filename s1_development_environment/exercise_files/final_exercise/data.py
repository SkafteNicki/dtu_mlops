import torch
import numpy as np
from torchvision import datasets, transforms


def mnist():
    # exchange with the corrupted mnist dataset
    # Load 4 train_ npz files
    X_train = np.load('dtu_mlops/data/corruptmnist/train_0.npz')['images']
    y_train = np.load('dtu_mlops/data/corruptmnist/train_0.npz')['labels']

    for i in range(1, 5):
        X_train = np.concatenate((X_train, np.load(f'dtu_mlops/data/corruptmnist/train_{i}.npz')['images']), axis=0)
        y_train = np.concatenate((y_train, np.load(f'dtu_mlops/data/corruptmnist/train_{i}.npz')['labels']), axis=0)
        
    
    X_test = np.load('dtu_mlops/data/corruptmnist/test.npz')["images"]
    y_test = np.load('dtu_mlops/data/corruptmnist/test.npz')["labels"]

    # Add labels to the test set
    test = torch.utils.data.TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())

    # Add labels to the train set
    train = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())

    return train, test

if __name__ == "__main__":
    train, test = mnist()
    print(train[0])
    print(test[0])
    print(len(train))
    print(len(test))
