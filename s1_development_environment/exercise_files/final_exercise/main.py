import argparse
import sys

import torch
from torch.autograd import Variable
from torch import optim
import torch.nn as nn
import click

from data import mnist
from model import MyAwesomeModel


@click.group()
def cli():
    pass


@click.command()
@click.option("--lr", default=1e-3, help='learning rate to use for training')
@click.option("--num_epochs", default=100, help='epochs to use for training')
def train(num_epochs, lr):
    print("Training day and night")
    print(lr)
    print(num_epochs)

    # TODO: Implement training loop here
    model = MyAwesomeModel().double()
    train_set, _ = mnist() # train_set['images'], train_set['labels']
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.01)
    model.train()
        
    for epoch in range(num_epochs):
        total_steps = len(train_set['images'])
        for i in range(total_steps):
            image = torch.from_numpy(train_set['images'][i].reshape((1, 1, 28, 28)))
            label = torch.from_numpy(train_set['labels'][i])
            
            # gives batch data, normalize x when iterate train_loader
            x = Variable(image)   # batch x
            y = Variable(label)   # batch y
            
            output = model(x.double())[0]
            loss = loss_func(output, y)
            
            # clear gradients for this training step   
            optimizer.zero_grad()
            
            # backpropagation, compute gradients
            loss.backward()
            # apply gradients
            optimizer.step()

            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_steps}], Loss: {loss.item()}')


@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = torch.load(model_checkpoint)
    _, test_set = mnist()


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()


    
    
    
    