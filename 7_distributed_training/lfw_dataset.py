"""
LFW dataloading
"""
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import argparse
import time
import numpy as np

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
        transforms.RandomAffine(5, (10, 10), (0.5, 2.0)),
        transforms.ToTensor()
    ])
    
    # Define dataset
    dataset = LFWDataset(args.path_to_folder, lfw_trans)
    
    # Define dataloader
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False,
                            num_workers=args.num_workers)
    
    if args.visualize_batch:
        # TODO: visualize a batch of images
        pass
        
    if args.timing:
        # lets do so repetitions
        res = [ ]
        for _ in range(5):
            start = time.time()
            for batch in dataloader:
                # similate that we do something with the batch
                time.pause(0.2)
            end = time.time()
            
            res.append(end - start)
            
        res = np.array(res)
        print('Timing: {np.mean(res)}+-{np.std(res)}')
        