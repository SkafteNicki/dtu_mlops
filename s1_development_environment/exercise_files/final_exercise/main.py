import argparse
import sys

import torch
import click

from data import mnist
from model import MyAwesomeModel

from torch import nn, optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

@click.group()
def cli():
    pass


@click.command()
@click.option("--lr", default=1e-3, help='learning rate to use for training')
def train(lr):
    print("Training day and night")
    print(lr)

    model = MyAwesomeModel()
    train_set, _ = mnist()
    trainloader = DataLoader(train_set, batch_size=64, shuffle=True)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    epochs = 200
    epochs_loss = []
    for _ in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            # Flatten MNIST images into a 784 long vector
            images = images.view(images.shape[0], -1)
            optimizer.zero_grad() # reset gradients
            output = model(images.float()) # run data through model
            loss = criterion(output, labels) # calculate loss
            loss.backward() # calculate gradient
            optimizer.step() # back probagate
            running_loss += loss.item() # calculate loss
        else:
            loss_epoch_mean = running_loss/len(trainloader)
            epochs_loss.append(loss_epoch_mean)
            print(f"Training loss: {running_loss/len(trainloader)}")


    plt.plot(epochs_loss)
    plt.show()
    torch.save(model.state_dict(), "model_checkpoint.pt")

@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    model = MyAwesomeModel()
    model.load_state_dict(torch.load("model_checkpoint.pt"))

    _, test_set = mnist()
    testloader = DataLoader(test_set, batch_size=len(test_set))
    
    model.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images = images.view(images.shape[0], -1)
            ps = model(images.float())
            _, top_class = ps.topk(1, dim=1)
            top_class = torch.squeeze(top_class)
            equals = top_class == labels.view(*top_class.shape)
            accuracy = torch.mean(equals.type(torch.FloatTensor))
            print(f"Accuracy: {accuracy.item()*100}%")          


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()


    
    
    
    