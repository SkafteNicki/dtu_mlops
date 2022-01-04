import argparse
import os
import sys

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from pathlib import Path

import numpy as np
from data import mnist
from model import MyAwesomeModel


class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>"
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')

            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()

    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--lr', default=0.1)
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)

        # hyperparameter definition
        model = MyAwesomeModel()
        train_set, _ = mnist()
        criterion = nn.NLLLoss()
        lr = 0.001
        optimizer = optim.Adam(model.parameters(), lr=lr)
        epochs = 20
        batch_size = 64
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

        for e in range(epochs):
            running_loss = 0
            acc = []
            for i, data in enumerate(train_loader):
                optimizer.zero_grad()
                img, labels = data
                x = img.unsqueeze(1)
                output = model(x.float())
                ps = torch.exp(output)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                acc.append(torch.mean(equals.type(torch.FloatTensor)).item())
                loss = criterion(output, labels)
                loss.backward()
                running_loss += loss.item()

                optimizer.step()
            else:
                print(f"Epoch {e+1} --> Loss: {running_loss:.6} Accuracy: {np.sum(acc)/len(acc)*100:.4}%")

        i = 0
        while True:
            if not os.path.isfile(f"models/model_e{epochs}_b{batch_size}_lr{lr}_{i}.pth"):
                break
            else:
                i += 1
        torch.save(model, f"models/model_e{epochs}_b{batch_size}_lr{lr}_{i}.pth")

    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('load_model_from', default="")
        args = parser.parse_args(sys.argv[2:])

        model = torch.load(args.load_model_from)

        _, test_set = mnist()
        test_loader = DataLoader(test_set, batch_size=64, shuffle=True)
        with torch.no_grad():
            model.eval()
            acc = []
            for i, data in enumerate(test_loader):
                img, labels = data
                x = img.unsqueeze(1).float()
                output = model(x)
                ps = torch.exp(output)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                acc.append(torch.mean(equals.type(torch.FloatTensor)).item())

            print(f"Final test accuracy: {np.sum(acc) / len(acc)*100:.4}%")


if __name__ == '__main__':
    TrainOREvaluate()
    
    
    
    
    
    
    
    
    