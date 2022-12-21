import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np


class FluidDataSet(Dataset):
    """ 
    Customized dataset for smoke simulation
    """
    def __init__(self, root: str, dataset_name: str, normalized: bool=True) -> None:
        """
        The individual dataset should be the following format:
        self.density: (N, 1, H, W)
        self.velocity: (N, 2, H, W) (NOTE: for the 2nd dimension, the 1st one is the y conponent, the 2nd one is the x component)
        self.param: (N, 4) (time, normalized time, source_pos x, source_pos y)
        """
        self.data = np.load(root+dataset_name+'.npz', allow_pickle=True)
        self.density = torch.from_numpy(self.data['density']).float().unsqueeze(1)
        self.velocity = torch.from_numpy(self.data['velocity']).float().permute(0, 3, 1, 2)
        self.param = torch.from_numpy(self.data['param']).float()
        self.transform = transforms.Compose([
            transforms.Normalize(mean=[0.5], std=[0.5]) if normalized else transforms.Lambda(lambda x: x)
        ])
        self.density = self.transform(self.density)
        self.max_time = self.param[:, 0].max().item()

    def __getitem__(self, index: int) -> tuple:
        rho = self.density[index]
        vel = self.velocity[index]
        param = self.param[index]
        return (rho, vel, param)
    
    def __len__(self) -> int:
        return self.density.shape[0]
    



    