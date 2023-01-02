import numpy as np
import torch
from torch.utils.data import DataLoader


def mnist():
    train = dict()
    temp0 = np.load(f"/Users/Gustav/Desktop/DesktopV00183/DTU/1st/MLOps.nosync/dtu_mlops/data/corruptmnist/train_0.npz")['images']
    temp1 = np.load(f"/Users/Gustav/Desktop/DesktopV00183/DTU/1st/MLOps.nosync/dtu_mlops/data/corruptmnist/train_1.npz")['images']
    temp2 = np.load(f"/Users/Gustav/Desktop/DesktopV00183/DTU/1st/MLOps.nosync/dtu_mlops/data/corruptmnist/train_2.npz")['images']
    temp3 = np.load(f"/Users/Gustav/Desktop/DesktopV00183/DTU/1st/MLOps.nosync/dtu_mlops/data/corruptmnist/train_3.npz")['images']
    temp4 = np.load(f"/Users/Gustav/Desktop/DesktopV00183/DTU/1st/MLOps.nosync/dtu_mlops/data/corruptmnist/train_4.npz")['images']
    temp_data = np.concatenate((temp0, temp1, temp2, temp3, temp4))
    temp_0 = np.load(f"/Users/Gustav/Desktop/DesktopV00183/DTU/1st/MLOps.nosync/dtu_mlops/data/corruptmnist/train_0.npz")['labels']
    temp_1 = np.load(f"/Users/Gustav/Desktop/DesktopV00183/DTU/1st/MLOps.nosync/dtu_mlops/data/corruptmnist/train_1.npz")['labels']
    temp_2 = np.load(f"/Users/Gustav/Desktop/DesktopV00183/DTU/1st/MLOps.nosync/dtu_mlops/data/corruptmnist/train_2.npz")['labels']
    temp_3 = np.load(f"/Users/Gustav/Desktop/DesktopV00183/DTU/1st/MLOps.nosync/dtu_mlops/data/corruptmnist/train_3.npz")['labels']
    temp_4 = np.load(f"/Users/Gustav/Desktop/DesktopV00183/DTU/1st/MLOps.nosync/dtu_mlops/data/corruptmnist/train_4.npz")['labels']
    temp_labels = np.concatenate((temp_0, temp_1, temp_2, temp_3, temp_4))
    train['images'] = temp_data
    train['labels'] = temp_labels

    test = np.load("/Users/Gustav/Desktop/DesktopV00183/DTU/1st/MLOps.nosync/dtu_mlops/data/corruptmnist/test.npz")

    """
    loaders = {
    'train' : torch.utils.data.DataLoader(train, 
                                          batch_size=100, 
                                          shuffle=True, 
                                          num_workers=1),
    
    'test'  : torch.utils.data.DataLoader(test, 
                                          batch_size=100, 
                                          shuffle=True, 
                                          num_workers=1),
    }
    print(loaders)
    for i, (images, labels) in enumerate(loaders['train']):
        print(images, labels)
        break
    """
    return train, test

if __name__ == '__main__':
    mnist()
