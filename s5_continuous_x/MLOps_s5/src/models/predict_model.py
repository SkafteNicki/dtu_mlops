# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
# from dotenv import find_dotenv, load_dotenv
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from src.models.model import MyAwesomeConvolutionalModel  # MyAwesomeModel


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
def main(input_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    # load model
    model = MyAwesomeConvolutionalModel()
    model.load_state_dict(torch.load(input_filepath + "/trained_model.pt"))
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    # load test data
    test_images = torch.load("data/processed/test_images_tensor.pt")
    test_labels = torch.load("data/processed/test_labels_tensor.pt")
    test_dataset = Dataset(test_images, test_labels)

    # print("hej")
    test_dataloader = torch.utils.data.DataLoader(test_dataset, shuffle=True)
    test_losses = []
    test_accuracy = []
    with torch.no_grad():
        # model.eval

        running_loss_test = 0
        running_accuracy_test = 0

        for images, labels in test_dataloader:
            log_ps = model(images.float())

            # loss
            # print(labels.shape)

            loss = criterion(log_ps, labels)
            running_loss_test += loss.item()

            # accuracy
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy = torch.mean(equals.type(torch.FloatTensor))
            running_accuracy_test += accuracy.item()
            # print(accuracy.item())

        # save
        test_losses.append(running_loss_test / len(test_dataloader))
        test_accuracy.append(running_accuracy_test / len(test_dataloader))

        print(test_accuracy)
        print(test_losses)


class Dataset(torch.utils.data.Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(self, images, labels):
        "Initialization"
        self.labels = labels
        self.images = images

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.images)

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        X = self.images[index]

        # Load data and get label
        # X = torch.load('data/' + ID + '.pt')
        y = self.labels[index]

        return X, y


if __name__ == "__main__":
    # log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    # logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    # project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())

    main()
