import numpy as np
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, path):
        if type(path) == str:
            file = np.load(path)
            self.images = file['images']
            self.labels = file['labels']
        elif type(path) == list:
            self.images = []
            self.labels = []
            for p in path:
                file = np.load(p)
                self.images.append(file['images'])
                self.labels.append(file['labels'])
            self.images = np.concatenate(self.images)
            self.labels = np.concatenate(self.labels)
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]


if __name__=='__main__':
    testdata = CustomDataset('S1/mnist/corruptmnist/test.npz')
    traindata = CustomDataset(['S1/mnist/corruptmnist/train_0.npz', 'S1/mnist/corruptmnist/train_1.npz', 'S1/mnist/corruptmnist/train_2.npz', 'S1/mnist/corruptmnist/train_3.npz', 'S1/mnist/corruptmnist/train_4.npz'])

    testloader = DataLoader(testdata, batch_size=64, shuffle=True)
    trainloader = DataLoader(traindata, batch_size=64, shuffle=True)

    import matplotlib.pyplot as plt
    img, label = next(iter(testloader))
    for i in range(10):
        plt.imshow(img[i], cmap="gray")
        plt.show()

