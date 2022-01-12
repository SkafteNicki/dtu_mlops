"""
LFW dataloading
"""
import argparse
import time
import os, sys

import os
import glob
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torchvision.transforms.functional as F



class LFWDataset(Dataset):
    def __init__(self, path_to_folder: str, transform) -> None:
        # TODO: fill out with what you need
        files = glob.glob('data/' + '/**/*.jpg', recursive=True)
        self.files_path = files
        self.path_to_folder = path_to_folder
        self.transform = transform
        
    def __len__(self):
        files_path = self.files_path
        return len(files_path) # TODO: fill out
    
    def __getitem__(self, index: int) -> torch.Tensor:
        # TODO: fill out
        files_path = self.files_path
        img = Image.open(files_path[index])

        return self.transform(img)


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()

       
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-path_to_folder', default='/data/lfw-deepfunneled', type=str) 
    parser.add_argument('-batch_size', default=512, type=int)
    parser.add_argument('-num_workers', default=4, type=int)
    parser.add_argument('-visualize_batch', action='store_true')
    parser.add_argument('-get_timing', action='store_true')
    parser.add_argument('-batches_to_check', default=100, type=int)
    
    args = parser.parse_args()
    
    lfw_trans = transforms.Compose([
        transforms.RandomAffine(5, (0.1, 0.1), (0.5, 2.0)),
        transforms.ToTensor()
    ])
    
    # Define dataset
    dataset = LFWDataset(args.path_to_folder, lfw_trans)
    
    # Define dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers
    )
    
    if args.visualize_batch:
        # TODO: visualize a batch of images
        batch = next(iter(dataloader))
        img = make_grid(batch)
        show(img)
            
            
        
        
    if args.get_timing:
        # lets do some repetitions
        res = [ ]
        for _ in range(5):
            start = time.time()
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx > args.batches_to_check:
                    break
            end = time.time()

            res.append(end - start)
            
        res = np.array(res)
        #m = np.mean(res)
        #s = np.std(res)
        print('Timing:' + str(np.mean(res)) + '+-' + str(np.std(res)))
        #print('Timing: np.mean(res)+-np.std(res)')

