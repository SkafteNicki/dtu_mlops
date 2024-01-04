import torch


def mnist():
    """Return train and test dataloaders for MNIST."""
    train_data, train_labels = [ ], [ ]
    for i in range(5):
        train_data.append(torch.load(f"/Users/diogoadegas/Documents/Machine_Learning_Operations/dtu_mlops/data/corruptmnist/train_images_{i}.pt"))
        train_labels.append(torch.load(f"/Users/diogoadegas/Documents/Machine_Learning_Operations/dtu_mlops/data/corruptmnist/train_target_{i}.pt"))

    train_data = torch.cat(train_data, dim=0)
    train_labels = torch.cat(train_labels, dim=0)

    test_data = torch.load("/Users/diogoadegas/Documents/Machine_Learning_Operations/dtu_mlops/data/corruptmnist/test_images.pt")
    test_labels = torch.load("/Users/diogoadegas/Documents/Machine_Learning_Operations/dtu_mlops/data/corruptmnist/test_target.pt")

    print(train_data.shape)
    print(train_labels.shape)
    print(test_data.shape)
    print(test_labels.shape)

    train_data = train_data.unsqueeze(1)
    test_data = test_data.unsqueeze(1)

    return (
        torch.utils.data.TensorDataset(train_data, train_labels), 
        torch.utils.data.TensorDataset(test_data, test_labels)
    )