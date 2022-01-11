import argparse
import sys

import click
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

# for logging
import wandb

#from data import mnist
from src.models.model import MyAwesomeConvolutionalModel  # MyAwesomeModel


@click.command()
@click.argument('lr_1', type=float)
@click.argument('epochs_1', type=int)
def main(lr_1,epochs_1):
        print("Training day and night")
        
        # TODO: Implement training loop here
        model = MyAwesomeConvolutionalModel()

        train_images = torch.load('data/processed/train_images_tensor.pt')
        train_labels = torch.load('data/processed/train_labels_tensor.pt')

        train_dataset = Dataset(train_images, train_labels)
        
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr = lr_1)#float(args['lr']))#lr=0.003)

        train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=64, shuffle=True)

        epochs = epochs_1 
        train_losses = []
        train_accuracy = []
        for e in range(epochs):
            print(e)
            running_loss_train = 0
            running_accuracy_train = 0
            for images, labels in train_dataloader: 
                optimizer.zero_grad()
                    
                log_ps = model(images.float())

                # loss
                loss = criterion(log_ps, labels)
                loss.backward()
                optimizer.step()
                running_loss_train += loss.item()

                # accuray
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy = torch.mean(equals.type(torch.FloatTensor))
                running_accuracy_train += accuracy.item()

            # log loss
            wandb.log({'loss': running_loss_train/len(train_dataloader)})
            
            train_losses.append(running_loss_train/len(train_dataloader))
            train_accuracy.append(running_accuracy_train/len(train_dataloader))

            print(running_loss_train/len(train_dataloader))

        # save plots
        plt.figure(1)
        plt.plot(range(epochs),train_losses)
        plt.savefig('reports/figures/train_loss.png')
        plt.figure(2)
        plt.plot(range(epochs),train_accuracy)
        wandb.log({"Training_Accuracy_1": plt})
        plt.savefig('reports/figures/train_accuracy.png')
        
        # save model
        torch.save(model.state_dict(), 'models/trained_model.pt')


class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, images, labels):
        'Initialization'
        self.labels = labels
        self.images = images

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.images)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        X = self.images[index]

        # Load data and get label
        #X = torch.load('data/' + ID + '.pt')
        y = self.labels[index]

        return X, y


if __name__ == '__main__':
    # init logging
    wandb.init(project='testing-wandb') #entity

    main()