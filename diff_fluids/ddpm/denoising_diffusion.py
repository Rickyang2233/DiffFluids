from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

def gather(consts: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1, 1)

class DenoisingDiffusion:
    def __init__(self,
                 eps_model: nn.Module,
                 n_steps: int,
                 device: torch.device):
        super().__init__()
        self.eps_model = eps_model
        self.n_steps = n_steps
        self.device = device
        self.beta = torch.linspace(1e-4, 0.02, n_steps, dtype=torch.float32, device=device)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.sigma2 = self.beta
    

    def q_xt_x0(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple:
        mean = (gather(self.alpha_bar, t) ** 0.5) * x0
        var = 1 - gather(self.alpha_bar, t)
        return (mean, var)
    
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor]) -> torch.Tensor:
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps

    def p_xtminus1_xt(self, x_t: torch.Tensor, t: torch.Tensor) -> Tuple:
        var = gather(self.sigma2, t)

        eps_pred = self.eps_model(x_t, t)
        alpha_bar = gather(self.alpha_bar, t)
        alpha = gather(self.alpha, t)
        eps_coeff = (1 - alpha) / ((1 - alpha_bar) ** 0.5)
        mean = 1 / (alpha ** 0.5) * (x_t - eps_coeff * eps_pred)

        return (mean, var)
    
    def p_sample(self, x_t: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor]) -> torch.Tensor:
        if eps is None:
            eps = torch.randn_like(x_t)
        mean, var = self.p_xtminus1_xt(x_t, t)
        return mean + (var ** 0.5) * eps

    def ddpm_loss(self, x0: torch.Tensor, noise: Optional[torch.Tensor]=None, cond: torch.Tensor=None) -> torch.Tensor:
        
        batch_size = x0.shape[0]
        t = torch.randint(0, self.n_steps, (batch_size,), device=self.device, dtype=torch.long)

        if noise is None:
            noise = torch.randn_like(x0)
        
        x_t = self.q_sample(x0, t, eps=None)
        eps_pred = self.eps_model(x_t, t, cond)
        return F.mse_loss(noise, eps_pred)
        
        