import argparse
import sys

import torch
import click

from data import mnist
from model import MyAwesomeModel
from torch.utils.data import DataLoader
from torch import nn, optim 
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch 

@click.group()
def cli():
    pass

@click.command()
@click.option("--lr", default=1e-3, help='learning rate to use for training')
def train(lr):
    print("Training day and night")
    print("lr: ", lr)

    # TODO: Implement training loop here
    model = MyAwesomeModel()
    train_set, _ = mnist()
    trainloader = DataLoader(train_set, batch_size=5)

    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr)

    epochs = 100
    training_loss = []

    for _ in range(epochs):
        running_loss = 0
        for images, labels in trainloader:

            # Flatten MNIST images into a 784 long vector
            images = images.view(images.shape[0], -1)
            # Clear the gradients, do this because gradients are accumulated
            optimizer.zero_grad()
             
            output = model(images.float()) 
            loss = criterion(output, labels) 
            loss.backward() 
            optimizer.step() 
            running_loss += loss.item() 
        else:
            print(f"Training loss: {running_loss/len(trainloader)}")
            training_loss.append(running_loss/len(trainloader))

    # save model
    torch.save(model.state_dict(), 'model_checkpoint.pth')

    # plot loss curve
    plt.plot(training_loss)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()


@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    model = MyAwesomeModel()
    model.load_state_dict(torch.load(model_checkpoint))

    _, test_set = mnist()
    testloader = DataLoader(test_set,batch_size=len(test_set)) # the whole set

    model.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images = images.view(images.shape[0], -1)
            output = model(images.float())
            _, top_class = output.topk(1, dim=1)
            top_class = torch.squeeze(top_class)
            equals = top_class == labels.view(*top_class.shape)
            accuracy = torch.mean(equals.type(torch.FloatTensor))
            print(f'Accuracy: {accuracy.item()*100}%')

cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()