import argparse
import sys

import torch

from torch import optim, nn
from model import MyAwesomeModel
from data import CustomDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


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
        
        # TODO: Implement training loop here
        model = MyAwesomeModel()
        traindata = CustomDataset(['corruptmnist/train_0.npz', 'corruptmnist/train_1.npz', 'corruptmnist/train_2.npz', 'corruptmnist/train_3.npz', 'corruptmnist/train_4.npz'])
        trainloader = DataLoader(traindata, batch_size=32, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.03)

        train_loss = []
        for e in range(10):
            running_loss = 0
            model.train()
            for images, labels in trainloader:
                images = images.view(images.shape[0], -1).float()
                optimizer.zero_grad()
                
                log_ps = model(images)
                loss = criterion(log_ps, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            train_loss.append(running_loss)
            print(f'Epoch {e:2}: {running_loss / len(trainloader)}')
        plt.plot(train_loss)
        plt.show()
        torch.save(model, 'trained_model.pt')
        
        
    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('load_model_from', default="")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement evaluation logic here
        model = torch.load(args.load_model_from)

        testdata = CustomDataset('corruptmnist/test.npz')
        testloader = DataLoader(testdata, batch_size=64, shuffle=True)
        model.eval()
        accuracy = 0
        with torch.no_grad():
            for images, labels in testloader:
                images = images.view(images.shape[0], -1).float()
                output = model(images)
                top_p, top_class = output.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy = torch.mean(equals.type(torch.FloatTensor))
            print(f'Accuracy: {accuracy.item()*100}%')

if __name__ == '__main__':
    TrainOREvaluate()