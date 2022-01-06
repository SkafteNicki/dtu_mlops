# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import click
import numpy as np
#from dotenv import find_dotenv, load_dotenv
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    #print("hej")
    train_images, train_labels, test_images, test_labels = mnist(input_filepath)

    torch.save(train_images, output_filepath + '/train_images_tensor.pt')
    torch.save(train_labels, output_filepath + '/train_labels_tensor.pt')
    torch.save(test_images, output_filepath + '/test_images_tensor.pt')
    torch.save(test_labels, output_filepath + '/test_labels_tensor.pt')




def mnist(input_filepath):
      # exchange with the corrupted mnist dataset
      #train = torch.randn(50000, 784)
      #test = torch.randn(10000, 784) 

      #train
      train0 = np.load(input_filepath + "/train_0.npz")
      train1 = np.load(input_filepath + "/train_1.npz")
      train2 = np.load(input_filepath + "/train_2.npz")
      train3 = np.load(input_filepath + "/train_3.npz")
      train4 = np.load(input_filepath + "/train_4.npz")
      train_images = torch.cat((torch.from_numpy(train0.f.images),torch.from_numpy(train1.f.images),torch.from_numpy(train2.f.images),torch.from_numpy(train3.f.images),torch.from_numpy(train4.f.images),), 0)
      train_labels = torch.cat((torch.from_numpy(train0.f.labels),torch.from_numpy(train1.f.labels),torch.from_numpy(train2.f.labels),torch.from_numpy(train3.f.labels),torch.from_numpy(train4.f.labels),), 0)
      #train = Dataset(train_images, train_labels)

      #test
      test0 = np.load(input_filepath + "/test.npz")
      test_images = torch.from_numpy(test0.f.images)
      test_labels = torch.from_numpy(test0.f.labels)
      #test = Dataset(test_images, test_labels)

      transform = transforms.Normalize((0.5,), (0.5,))
      train_images = transform(train_images)
      test_images = transform(test_images)

      
      print(train_images.shape)
      #print(train_images[0])

      train_images = train_images.reshape((-1, 1, 28, 28))
      test_images = test_images.reshape((-1, 1, 28, 28))
      print(train_images.shape)

      #train = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
      

      return train_images, train_labels, test_images, test_labels



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    #load_dotenv(find_dotenv())
    
    main()


