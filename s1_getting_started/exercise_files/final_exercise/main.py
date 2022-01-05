import argparse
import sys

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

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
        
        # TODO: Implement training loop here
        model = MyAwesomeModel()
        train_dataset, _ = mnist()
        #images = train_set['images']
        #images = images.view(images.shape[0], -1)
        #labels = train_set['labels']

        #print(images.shape)
        #tmp = train_set['images']
        
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.003)

        train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=64, shuffle=True)


        epochs = 30
        steps = 0
        train_losses = []
        train_accuracy = []
        #train_losses, test_losses = [], []
        for e in range(epochs):
            running_loss_train = 0
            running_accuracy_train = 0
            for images, labels in train_dataloader: #batch
                #images = batch[0]
                #images = batch[1]
                
                optimizer.zero_grad()
                    
                log_ps = model(images.float())

                # loss
                loss = criterion(log_ps, labels)
                loss.backward()
                optimizer.step()
                #running_loss += loss.item()
                #train_losses.append(running_loss)
                running_loss_train += loss.item()

                # accuray
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy = torch.mean(equals.type(torch.FloatTensor))
                running_accuracy_train += accuracy.item()
                #train_accuracy.append(accuracy.item())
            
            train_losses.append(running_loss_train/len(train_dataloader))
            train_accuracy.append(running_accuracy_train/len(train_dataloader))

        #plt.plot(range(epochs),train_losses)
        #print(train_accuracy)

        torch.save(model.state_dict(), 'trained_model.pth')

        
        
    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('load_model_from', default="")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        ## TODO: Implement evaluation logic here

        # load model
        model = MyAwesomeModel()
        model.load_state_dict(torch.load(args.load_model_from))
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.003)

        # load test data
        _, test_dataset = mnist()


        test_dataloader = torch.utils.data.DataLoader(test_dataset, shuffle=True)
        test_losses = []
        test_accuracy = []
        with torch.no_grad():
            #model.eval
            
            running_loss_test = 0
            running_accuracy_test = 0
            
            for images, labels in test_dataloader:
                log_ps = model(images.float())
        
                # loss
                loss = criterion(log_ps, labels)
                running_loss_test += loss.item()
                
                # accuracy
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy = torch.mean(equals.type(torch.FloatTensor))
                running_accuracy_test += accuracy.item()
                #print(accuracy.item())

            # save
            test_losses.append(running_loss_test/len(test_dataloader))
            test_accuracy.append(running_accuracy_test/len(test_dataloader))
        
            print(test_accuracy)
            print(test_losses)
        

if __name__ == '__main__':
    TrainOREvaluate()
    
    
    
    
    
    
    
    
    