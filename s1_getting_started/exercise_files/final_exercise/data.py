import numpy as np
import torch
import os



def mnist():

    train_images = []
    train_labels = []
    PARENTPath = '/Users/johannespischinger/Documents/Uni/Master/Erasmus/Courses/MLOps/dtu_mlops'
    DATAPath = '/data/corruptmnist/'

    for idx in range(5):
        train_data = np.load(PARENTPath+DATAPath+'train_{}.npz'.format(idx))
        train_images.append(train_data['images'])
        train_labels.append(train_data['labels'])

    train_images = torch.from_numpy(np.asarray(train_images))
    train_labels = torch.from_numpy(np.asarray(train_labels))

    train_images = train_images.view(5*5000,28,28)
    train_labels = train_labels.view(5 * 5000, -1).flatten()

    train = torch.utils.data.TensorDataset(train_images,train_labels)

    test_data = np.load(PARENTPath+DATAPath+"test.npz".format(idx))
    test_images = test_data['images']
    test_labels = test_data['labels']

    test_images = torch.from_numpy(np.asarray(test_images))
    test_labels = torch.from_numpy(np.asarray(test_labels))
    test_images = test_images.view(5000, 28, 28)
    test_labels = test_labels.view(5000, -1).flatten()

    test = torch.utils.data.TensorDataset(test_images,test_labels)


    return train, test


train,test = mnist()