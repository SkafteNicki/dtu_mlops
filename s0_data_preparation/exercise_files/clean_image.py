# from this tutorial:
# https://docs.cleanlab.ai/stable/tutorials/image.html
from torch.utils.data import DataLoader, TensorDataset, Subset
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import StratifiedKFold
import numpy as np
import matplotlib.pyplot as plt

from tqdm.autonotebook import tqdm
import math
import time
import multiprocessing

from cleanlab import Datalab
from datasets import load_dataset

dataset = load_dataset("fashion_mnist", split="train")
print(dataset)

num_classes = len(dataset.features["label"].names)
print(num_classes)

transformed_dataset = dataset.with_format("torch")

# Apply transformations
def normalize(example):
    example["image"] = (example["image"] / 255.0).unsqueeze(0)
    return example

transformed_dataset = transformed_dataset.map(normalize, num_proc=multiprocessing.cpu_count())

torch_dataset = TensorDataset(transformed_dataset["image"], transformed_dataset["label"])

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.ReLU(),
            nn.BatchNorm2d(6),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, 2),
        )
        self.linear = nn.Sequential(nn.LazyLinear(128), nn.ReLU())
        self.output = nn.Sequential(nn.Linear(128, num_classes))

    def forward(self, x):
        x = self.embeddings(x)
        x = self.output(x)
        return x

    def embeddings(self, x):
        x = self.cnn(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.linear(x)
        return x
    
K = 3  # Number of cross-validation folds. Set to small value here to ensure quick runtimes, we recommend 5 or 10 in practice for more accurate estimates.
n_epochs = 2  # Number of epochs to train model for. Set to a small value here for quick runtime, you should use a larger value in practice.
patience = 2  # Parameter for early stopping. If the validation accuracy does not improve for this many epochs, training will stop.
train_batch_size = 64  # Batch size for training
test_batch_size = 512  # Batch size for testing
num_workers = multiprocessing.cpu_count()  # Number of workers for data loaders

# Create k splits of the dataset
kfold = StratifiedKFold(n_splits=K, shuffle=True, random_state=0)
splits = kfold.split(transformed_dataset, transformed_dataset["label"])

train_id_list, test_id_list = [], []

for fold, (train_ids, test_ids) in enumerate(splits):
    train_id_list.append(train_ids)
    test_id_list.append(test_ids)


pred_probs_list, embeddings_list = [], []
embeddings_model = None

for i in range(K):
    print(f"\nTraining on fold: {i+1} ...")

    # Create train and test sets and corresponding dataloaders
    trainset = Subset(torch_dataset, train_id_list[i])
    testset = Subset(torch_dataset, test_id_list[i])

    trainloader = DataLoader(
        trainset,
        batch_size=train_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    testloader = DataLoader(
        testset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    # Train model
    model = train(trainloader, testloader, n_epochs, patience)
    if embeddings_model is None:
        embeddings_model = model

    # Compute out-of-sample embeddings
    print("Computing feature embeddings ...")
    fold_embeddings = compute_embeddings(embeddings_model, testloader)
    embeddings_list.append(fold_embeddings)

    print("Computing predicted probabilities ...")
    # Compute out-of-sample predicted probabilities
    fold_pred_probs = compute_pred_probs(model, testloader)
    pred_probs_list.append(fold_pred_probs)

print("Finished Training")


# Combine embeddings and predicted probabilities from each fold
features = torch.vstack(embeddings_list).numpy()

logits = torch.vstack(pred_probs_list)
pred_probs = nn.Softmax(dim=1)(logits).numpy()


