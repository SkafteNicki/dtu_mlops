import argparse
import sys

import torch
import click

from data import mnist
from model import MyAwesomeModel

import matplotlib.pyplot as plt
import numpy as np
from torch import nn, optim

@click.group()
def cli():
    pass


@click.command()
@click.option("--lr", default=1e-3, help='learning rate to use for training')
@click.option("--epochs", default=7, help='number of epochs to train for')
def train(lr, epochs):
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    model = MyAwesomeModel()
    train_set, _ = mnist()
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    
    model.train()

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    training_loss = []

    for e in range(epochs):
        print(f"Epoch {e+1}/{epochs}")
        running_loss = 0
        for images, labels in trainloader:
            
            optimizer.zero_grad()
            
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        else:
            training_loss.append(running_loss/len(trainloader))
            print(f"Training loss: {training_loss[-1]}")

    # plot the training loss
    plt.plot(training_loss, label='Training loss')
    plt.legend(frameon=False)
    plt.show()
    print("Saving model as model_checkpoint.pth")
    print("Path: dtu_mlops/s1_development_environment/exercise_files/final_exercise/model_checkpoint.pth")
    torch.save(model, 'dtu_mlops/s1_development_environment/exercise_files/final_exercise/model_checkpoint.pth')


@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = torch.load("dtu_mlops/s1_development_environment/exercise_files/final_exercise/"+model_checkpoint)
    _, test_set = mnist()
    testloader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)

    model.eval()

    criterion = nn.NLLLoss()

    with torch.no_grad():
        accuracy = 0
        for images, labels in testloader:
    
            log_ps = model(images)

            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))
        
    print("Accuracy: ", accuracy.item()/len(testloader)*100)


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()



    
    
    
    