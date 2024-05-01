import torch.nn as nn

import torch
import math
import numpy as np

# https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/classifier_free_guidance.py
def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def linear_beta_schedule(timesteps: int) -> torch.Tensor:
    """Linear beta schedule.

    Args:
        timesteps: Number of timesteps.

    Returns:
        Beta coefficients with linear schedule.
    """
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

def get_1d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height
    return:
    pos_embed: [grid_size, embed_dim] 
    """
    grid = np.expand_dims(np.arange(grid_size, dtype=np.float32), axis = 0)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float16)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

class DDIM(nn.Module):
    def __init__(self, model, max_steps, img_size=28, device='cuda'):
        super(DDIM, self).__init__()

        self.noise_predictor = model
        self.device = device
        self.img_size = img_size
        self.max_steps = max_steps

        # get beta samp schedule 
        # betas is variance of gaussian noise at tstep
        betas = cosine_beta_schedule(max_steps)
        alphas = 1.0 - betas
        # the actual quantity used in each reverse step
        self.alphas_cumprod = torch.cumprod(alphas, axis=0)

        self.alphas_cumprod_prev = nn.functional.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        # sqrt(alpha_bar) and sqrt(1 - alpha_bar)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
    
    def q_sample(self, x, t, noise):
        #alpha = self.alphas_cumprod[t]
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        return sqrt_alphas_cumprod_t * x + sqrt_one_minus_alphas_cumprod_t * noise
    
    @torch.no_grad()
    def ddim_sample(self, cond, shape=[1,1,28,28], steps=None):
        # https://arxiv.org/pdf/2010.02502
        batch = cond.shape[0]
        device = self.device
        total_timesteps = self.max_steps
        sampling_timesteps = self.max_steps if not steps else steps

        times = torch.linspace(0, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        if shape is None: shape = [batch, 1, self.img_size, self.img_size]
        
        x = torch.randn(shape, device=device)

        for time, time_next in time_pairs:
            t = torch.full((batch,), time, device=device, dtype=torch.long).to(device)
            t_n = t - (time - time_next)

            e = self.noise_predictor(x, t, cond).squeeze(0)
            
            sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
            sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x.shape)
            

            x_0 = (x - sqrt_one_minus_alphas_cumprod_t * e) / sqrt_alphas_cumprod_t # eq 9

            if time_next < 0: return x_0

            sqrt_alphas_cumprod_t_n = torch.sqrt(extract(self.alphas_cumprod_prev, t_n, x.shape))
            sqrt_one_minus_alphas_cumprod_t_n = torch.sqrt(1 - extract(self.alphas_cumprod_prev, t_n, x.shape))

            x = sqrt_alphas_cumprod_t_n * x_0 + sqrt_one_minus_alphas_cumprod_t_n * e  # eq 12

        return x