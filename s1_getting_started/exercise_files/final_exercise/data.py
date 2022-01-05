import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


def mnist():
      # exchange with the corrupted mnist dataset
      #train = torch.randn(50000, 784)
      #test = torch.randn(10000, 784) 

      #train
      train0 = np.load("train_0.npz")
      train1 = np.load("train_1.npz")
      train2 = np.load("train_2.npz")
      train3 = np.load("train_3.npz")
      train4 = np.load("train_4.npz")
      train_images = torch.cat((torch.from_numpy(train0.f.images),torch.from_numpy(train1.f.images),torch.from_numpy(train2.f.images),torch.from_numpy(train3.f.images),torch.from_numpy(train4.f.images),), 0)
      train_labels = torch.cat((torch.from_numpy(train0.f.labels),torch.from_numpy(train1.f.labels),torch.from_numpy(train2.f.labels),torch.from_numpy(train3.f.labels),torch.from_numpy(train4.f.labels),), 0)
      train = Dataset(train_images, train_labels)

      #test
      test0 = np.load("test.npz")
      test_images = torch.from_numpy(test0.f.images)
      test_labels = torch.from_numpy(test0.f.labels)
      test = Dataset(test_images, test_labels)

      #train = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
      

      return train, test


class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, images, labels):
        'Initialization'
        self.labels = labels
        self.images = images

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.images)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        X = self.images[index]

        # Load data and get label
        #X = torch.load('data/' + ID + '.pt')
        y = self.labels[index]

        return X, y