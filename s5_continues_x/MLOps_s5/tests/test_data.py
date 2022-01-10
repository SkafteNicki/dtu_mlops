# test data
import torch
from src.data import make_dataset


def test_data_length(train_images,test_images):
    #dataset_train = torch.load('data/processed/train_images_tensor.pt')
    #dataset_test = torch.load('data/processed/test_images_tensor.pt')
    N_train = 25000
    N_test = 5000
    assert len(train_images) == N_train 
    assert len(test_images) == N_test


def test_torch_sizes(train_images,test_images):
    torch_size = torch.ones((1,28,28))
    #dataset_train = torch.load('data/processed/train_images_tensor.pt')
    assert [x.shape == torch_size for x in train_images]
    assert [x.shape == torch_size for x in test_images]


def test_labels_represented(train_labels,test_labels):
    #dataset_train = torch.load('data/processed/train_labels_tensor.pt')
    train_unique = torch.unique(train_labels)
    test_unique = torch.unique(test_labels)
    torch_unique = torch.tensor([0,1,2,3,4,5,6,7,8,9])
    assert torch.equal(train_unique, torch_unique)
    assert torch.equal(train_unique, torch_unique)




# main
if __name__ == '__main__':
    train_images, train_labels, test_images, test_labels = make_dataset.mnist("data/raw")

    print("Test 1")
    test_data_length(train_images,test_images)
    
    print("Test 2")
    test_torch_sizes(train_images,test_images)

    print("Test 3")
    test_labels_represented(train_labels,test_labels)


