import torch
import numpy as np

def mnist():
    # open data from npz file
    test_ims = torch.tensor(np.load(f'/Users/annabzinkowska/DTU/MLops/dtu_mlops/data/corruptmnist/test.npz')['images'])
    test_labels = torch.tensor(np.load(f"/Users/annabzinkowska/DTU/MLops/dtu_mlops/data/corruptmnist/test.npz")["labels"])
    train_ims = torch.tensor(np.array([np.load(f"/Users/annabzinkowska/DTU/MLops/dtu_mlops/data/corruptmnist/train_{i}.npz")["images"] for i in range(5)]).reshape(-1,28,28))
    train_labels = torch.tensor(np.array([np.load(f"/Users/annabzinkowska/DTU/MLops/dtu_mlops/data/corruptmnist/train_{i}.npz")["labels"] for i in range(5)]).reshape(-1))
    
    # exchange with the corrupted mnist dataset
    #train = torch.randn(50000, 784)
    #test = torch.randn(10000, 784) 
    return list(zip(train_ims, train_labels)), list(zip(test_ims, test_labels))


