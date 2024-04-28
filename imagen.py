import torch
import torch.nn as nn
import math
import random
import os
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
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

def see(x, label=None):
    x_dst = x[0].view(28, 28).cpu().numpy()  # Reshape and move to CPU
    plt.imshow(x_dst, cmap='gray')
    plt.title(f"Label: {'none' if not label else label}")
    plt.show()
    print(x_dst.sum())

class DDIM(nn.Module):
    def __init__(self, model, max_steps, img_size=28, device='cuda'):
        super(DDIM, self).__init__()

        self.noise_predictor = model
        self.device = device
        self.img_size = img_size
        self.max_steps = max_steps

        # get beta samp schedule 
        # betas is variance of gaussian noise at tstep
        betas = linear_beta_schedule(max_steps)
        alphas = 1.0 - betas
        # the actual quantity used in each reverse step
        self.alphas_cumprod = torch.cumprod(alphas, axis=0)

        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
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

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define hyperparameters
BATCH_SIZE = 128
EPOCHS = 6
TEST_INTERVAL = 6
LEARNING_RATE = 1e-3
SAVE_INTERVAL = 1
label2text = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}
# Define transforms
import tqdm
import torch.nn.functional as F
from torchvision.transforms import CenterCrop, Compose, Lambda, Resize, ToTensor
from dit import *

transform = Compose(
    [
        Resize(28),
        CenterCrop(28),
        ToTensor(),
        Lambda(lambda t: (t * 2) - 1),
    ]
)
# Load the Fashion MNIST dataset
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

img_size = 28
n_classes = 10
debug = 0
mode = 'train'
T = 300 
from_ckpt = 0

noise_predictor = DiT_B_4(
    input_size=img_size, 
    num_classes=10, 
    in_channels=1, 
    class_dropout_prob=0
)
diffusion = DDIM(noise_predictor, T, img_size=img_size).to(device)

if from_ckpt:
    state_dict = torch.load('model/ldm/epoch_2.pth')
    state_dict = {k.replace('noise_predictor.',''):v for k,v in state_dict.items()}
    noise_predictor.load_state_dict(state_dict)
#noise_predictor = Unet(channels=1, dim_mults=(1, 2, 4), dim=28)
optimizer = optim.Adam(noise_predictor.parameters(), lr=1e-3)
def sample():
    if not os.path.exists('outs'): os.makedirs('outs')
    diffusion.eval()
    noise_predictor.eval()
    BS = 9
    for steps in range(0,T,30):
        print(f'sampling {steps}')
        cond_labels = torch.randint(0, n_classes, (BS,), device=device).long()

        shape = [BS,1,28,28]

        img = diffusion.ddim_sample(cond_labels, shape=shape, steps=steps)
        img = img.view(BS, 28, 28).cpu().numpy()  # Reshape and move to CPU
        
        # Save sampled images as a single 3x3 grid
        fig, axs = plt.subplots(3, 3, figsize=(9, 9))
        axs = axs.flatten()
        for i in range(BS):
            axs[i].imshow(img[i], cmap='gray')
            axs[i].set_title(label2text[cond_labels[i].item()])
            axs[i].axis('off')
        plt.tight_layout()
        plt.savefig(f'outs/steps={steps}.png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)  # Close the figure

#sample()

noise_predictor.train()
diffusion.train()
for epoch in range(EPOCHS):
    losses = []
    for batch_idx, (x, cond) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
        if batch_idx == len(train_loader)-1: break
        optimizer.zero_grad()
        B = x.shape[0]
        t = torch.randint(0, T, (B,), device=device).long()

        if debug: see(x)
        x,cond = x.to(device), cond.to(device)

        # One-hot encode cond
        #cond = F.one_hot(cond, num_classes=n_classes).long()
        true_noise = torch.randn_like(x)# noise at x_t
        x_t = diffusion.q_sample(x,t,true_noise)

        if debug: see(x_t, label=f'noise level')
        noise_pred = noise_predictor(x_t, t, cond)
        noise_pred = noise_pred.reshape(B, -1)
        true_noise = true_noise.reshape(B, -1)
        #print(true_noise.shape, noise_pred.shape)
        assert noise_pred.shape == true_noise.shape
        loss = F.smooth_l1_loss(true_noise, noise_pred)
        loss.backward()
        optimizer.step()

        if (batch_idx+1) % 10 == 0: 
            print(sum(losses[-10:])/10)
        losses.append(loss.item())
    
    # Print training progress
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {sum(losses)/len(losses):.4f}")

    # Save model weights
    if debug or (epoch+1) % SAVE_INTERVAL == 0:
        model_save_path = f'model/ldm/epoch_{epoch+1}.pth'
        if not os.path.exists('model/ldm'): os.makedirs('model/ldm')
        torch.save(diffusion.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")

    # Evaluate on test set every TEST_INTERVAL epochs
    if (epoch + 1) % TEST_INTERVAL == 0:
        sample()

