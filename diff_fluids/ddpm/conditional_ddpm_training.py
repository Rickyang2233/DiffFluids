import argparse
import logging
import math
from typing import Union

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

from utils import FluidDataSet
from unet import UNet, UNetXAttn
from denoising_diffusion import DenoisingDiffusion

matplotlib.use('Agg')

parse = argparse.ArgumentParser(description='Denoising Diffusion Training')

parse.add_argument('--model', type=str, default='unet', help='model to use, default: unet')
parse.add_argument('--epochs', type=int, default=20, help='number of epochs, default: 20')
parse.add_argument('--batch-size', type=int, default=16, help='batch size, default: 16')

parse.add_argument('--device', type=int, default=0, help='device to use (default 0)')
parse.add_argument('--debug', action='store_true', help='debug mode, default False')

class Configs:
    eps_model: Union[UNet, UNetXAttn]
    diffuser: DenoisingDiffusion
    in_channels: int=1
    out_channels: int=1
    channels: int=32
    channel_multpliers: list=[1, 2, 4, 8]
    n_res_blocks: int=2
    attention_levels: list=[0, 1, 2]
    n_heads: int=4
    transformer_layers: int=1
    n_steps: int=400
    lr: float=2e-4
    lrf: float=0.1
    dataset: FluidDataSet
    data_loader: DataLoader
    optimizer: torch.optim.Adam
    tb_writer: SummaryWriter
    data_root: str='/media/bamf-big/gefan/DiffFluids/data/smoke/'
    dataset: str='res64x64_dt1_t100_nsrc256'
    tb_writer_root: str='/media/bamf-big/gefan/DiffFluids/diff_fluids/ddpm/logs/'
    model_save_root: str='/media/bamf-big/gefan/DiffFluids/diff_fluids/ddpm/checkpoint/'
    def __init__(self, args):
        if args.debug:
            logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(asctime)s:%(message)s')
        else:
            logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(asctime)s:%(message)s')
            self.tb_writer = SummaryWriter(log_dir=self.tb_writer_root + self.dataset +'/')
        
        self.args = args

        if args.model == 'xunet':
            self.eps_model = UNetXAttn(
                in_channels = self.in_channels,
                out_channels = self.out_channels,
                channels = self.channels,
                channel_multpliers = self.channel_multpliers,
                n_res_blocks = self.n_res_blocks,
                attention_levels = self.attention_levels,
                n_heads = self.n_heads,
                transformer_layers = self.transformer_layers,
                cond_channels = 3
            ).cuda(args.device)
        elif args.model == 'unet':
            self.eps_model = UNet(
                in_channels = self.in_channels,
                out_channels = self.out_channels,
                channels = self.channels,
                channel_multpliers = self.channel_multpliers,
                n_res_blocks = self.n_res_blocks,
                attention_levels = self.attention_levels,
                n_heads = self.n_heads,
                cond_channels = 3
            ).cuda(args.device)
        else:
            raise NotImplementedError('Only UNet and UNetXAttn are supported now.')

        self.diffuser = DenoisingDiffusion(
            eps_model=self.eps_model,
            n_steps=self.n_steps,
            device = args.device
        ).cuda(args.device)

        self.dataset = FluidDataSet(self.data_root, self.dataset)
        self.dataloader = DataLoader(self.dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

        self.optimizer = torch.optim.Adam(self.diffuser.eps_model.parameters(), lr=self.lr)
        lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - self.lrf) + self.lrf
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lf)

        logging.info('Configs initialized')
    
    def train(self):
        for epoch in range(1 + self.args.epochs):
            self.diffuser.eps_model.train()
            cum_loss = 0
            pbar = tqdm(self.dataloader)
            for i, batch in enumerate(pbar):
                density = batch[0].cuda(self.args.device)
                cond = batch[-1][:, 1:].cuda(self.args.device)
                loss = self.diffuser.ddpm_loss(density, cond)
                cum_loss += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                pbar.set_description(f'Epoch [{epoch}/{self.args.epochs}] | Loss: {(cum_loss/(i+1)):.3f}')
                if not self.args.debug:
                    self.tb_writer.add_scalar('batch loss', loss.item(), epoch * len(self.dataloader) + i)
            self.scheduler.step()
        torch.save(self.diffuser.eps_model.state_dict(), self.model_save_root + f'{self.dataset}.pth')
        logging.info('Training finished')
        self.tb_writer.close()
                        
if __name__ == '__main__':
    args = parse.parse_args()
    configs = Configs(args)
    configs.train()




