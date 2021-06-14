"""
LFW dataloading
"""
import argparse
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class LFWDataset(Dataset):
    def __init__(self, path_to_folder: str, transform) -> None:
        # TODO: fill out with what you need
        self.transform = transform
        
    def __len__(self):
        return None # TODO: fill out
    
    def __getitem__(self, index: int) -> torch.Tensor:
        # TODO: fill out
        return self.transform(img)

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-path_to_folder', default='', type=str)
    parser.add_argument('-num_workers', default=None, type=int)
    parser.add_argument('-visualize_batch', action='store_true')
    parser.add_argument('-get_timing', action='store_true')
    args = parser.parse_args()
    
    lfw_trans = transforms.Compose([
        transforms.RandomAffine(5, (0.1, 0.1), (0.5, 2.0)),
        transforms.ToTensor()
    ])
    
    # Define dataset
    dataset = LFWDataset(args.path_to_folder, lfw_trans)
    
    # Define dataloader
    # Note we need a high batch size to see an effect of using many
    # number of workers
    dataloader = DataLoader(dataset, batch_size=512, shuffle=False,
                            num_workers=args.num_workers)
    
    if args.visualize_batch:
        # TODO: visualize a batch of images
        pass
        
    if args.get_timing:
        # lets do so repetitions
        res = [ ]
        for _ in range(5):
            start = time.time()
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx > 100:
                    break
            end = time.time()

            res.append(end - start)
            
        res = np.array(res)
        print('Timing: {np.mean(res)}+-{np.std(res)}')
