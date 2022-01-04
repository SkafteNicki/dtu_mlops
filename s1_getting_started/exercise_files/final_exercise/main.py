import argparse
import sys
import numpy as np
import torch

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
        train_set, _ = mnist()
        epoch = 5
        batchsize=64
        trainloader = torch.utils.data.DataLoader(train_set,batchsize,shuffle=True)

        optimizer = torch.optim.Adam(model.parameters(),lr=0.003)
        criterion = torch.nn.NLLLoss()

        for e in range(epoch):
            running_loss = 0
            acc = []
            for images,labels in trainloader:
                optimizer.zero_grad()

                images = images.unsqueeze(1)

                output = model(images.float())
                ps = torch.exp(output)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                acc.append(torch.mean(equals.type(torch.FloatTensor)))


                loss = criterion(output,labels)
                loss.backward()

                optimizer.step()

                running_loss += loss.item()



            else:
                print(f"Training loss: {running_loss / len(trainloader)}")
                print(f"Accuracy: {sum(acc)/len(acc)}")

        checkpoint = {'input_size': (64,1,28,28),
                      'output_size': 10,
                      'optimizer_state_dicts': optimizer.state_dict(),
                      'state_dict': model.state_dict()}

        torch.save(checkpoint, 'trained_model.pt')

    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('load_model_from', default="")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement evaluation logic here
        checkpoint = torch.load(args.load_model_from)
        model = MyAwesomeModel()
        model.load_state_dict(checkpoint['state_dict'])
        #!!!! Important to turn off dropout
        model.eval()
        _, test_set = mnist()
        batchsize=64
        epoch = 5
        testloader = torch.utils.data.DataLoader(test_set,batch_size=batchsize,shuffle=True)

        #!!!! Important to forbid gradient calculation
        with torch.no_grad():
            running_loss = 0
            test_acc = []
            for images,labels in testloader:
                images = images.unsqueeze(1)
                ps = torch.exp(model(images.float()))
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                test_acc.append(torch.mean(equals.type(torch.FloatTensor)))

            else:
                print(f"Accuracy: {sum(test_acc) / len(test_acc)}")



if __name__ == '__main__':
    TrainOREvaluate()
    
