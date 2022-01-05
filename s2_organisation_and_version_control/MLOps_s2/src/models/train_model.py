import argparse
import sys

import click
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

#from data import mnist
from src.models.model import MyAwesomeConvolutionalModel  # MyAwesomeModel


@click.command()
@click.argument('lr_1', type=float)
@click.argument('epochs_1', type=int)
def main(lr_1,epochs_1):
        print("Training day and night")
        #parser = argparse.ArgumentParser(description='Training arguments')
        #parser.add_argument('--lr', default=0.1)
        #parser.add_argument('--epochs', default=10)
        # add any additional argument that you want
        #args = parser.parse_args(sys.argv[1:])
        #print(args)
        #args = vars(args)
        #print(args)
        
        # TODO: Implement training loop here
        model = MyAwesomeConvolutionalModel()

        train_images = torch.load('data/processed/train_images_tensor.pt')
        train_labels = torch.load('data/processed/train_labels_tensor.pt')

        train_dataset = Dataset(train_images, train_labels)

        #train_dataset, _ = mnist()
        #images = train_set['images']
        #images = images.view(images.shape[0], -1)
        #labels = train_set['labels']

        #print(images.shape)
        #tmp = train_set['images']
        
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr = lr_1)#float(args['lr']))#lr=0.003)

        train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=64, shuffle=True)


        epochs = epochs_1 #int(args['epochs']) #30
        steps = 0
        train_losses = []
        train_accuracy = []
        #train_losses, test_losses = [], []
        for e in range(epochs):
            print(e)
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

            print(running_loss_train/len(train_dataloader))
        #plt.plot(range(epochs),train_losses)
        #print(train_losses)
        #print(train_accuracy)

        # save plots
        plt.figure(1)
        plt.plot(range(epochs),train_losses)
        plt.savefig('reports/figures/train_loss.png')
        plt.figure(2)
        plt.plot(range(epochs),train_accuracy)
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
    #log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    #logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    #project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    #load_dotenv(find_dotenv())
    
    main()