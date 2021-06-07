import sys
import argparse
import numpy as np
import torch
import helper
from data import mnist
from Classifier import MyAwesomeModel
#from Classifier import Classifier2
from torch import nn, optim

from torch import nn

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
        parser.add_argument('--lr', default=0.003)
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        model = MyAwesomeModel()
        #model = Classifier2
        train_set, _ = mnist()
        trainloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.003)

        epochs = 5
        steps = 0

        train_losses, test_losses = [], []
        for e in range(epochs):
            print("Starting epoch ", e+1)
            running_loss = 0
            for images, labels in trainloader:
                model.train()
                optimizer.zero_grad()
        
                log_ps = model(images)
                loss = criterion(log_ps, labels)
                loss.backward()
                optimizer.step()
                #print('loss: ', loss)
                running_loss += loss.item()
            print('Loss: ', running_loss/len(images))
        torch.save(model.state_dict(), 'checkpoint.pth')
                
        
    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--load_model_from', default="checkpoint.pth")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement evaluation logic here
        if args.load_model_from:
            model = MyAwesomeModel()
            model.load_state_dict(torch.load('checkpoint.pth'))
            #model = torch.load(args.load_model_from)
            print('Hej')
        _, test_set = mnist()
        testloader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)
        running_accuracy = []
        # turn off gradients
        with torch.no_grad():
    
            # set model to evaluation mode
            
          
             # validation pass here
            for images, labels in testloader:
                model.eval()
                # validation pass here
                output = model(images)
                #print(output.shape)
                ps = torch.exp(output)

                op_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy = torch.mean(equals.type(torch.FloatTensor))
                running_accuracy.append(accuracy)

            # Output of the network are log-probabilities, need to take exponential for probabilities    
            #helper.view_classify(images.view(1, 28, 28), ps)    
            print('Accuracy:' , np.mean(running_accuracy))

if __name__ == '__main__':
    TrainOREvaluate()
    
    