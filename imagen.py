import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from models.diffusion import DDIM

import torch
import os

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
from models.dit import DiT_B_4 as DiT

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

noise_predictor = DiT(
    input_size=img_size, 
    num_classes=10, 
    in_channels=1, 
    class_dropout_prob=0,
    learn_sigma=False
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

        x,cond = x.to(device), cond.to(device)

        # One-hot encode cond
        #cond = F.one_hot(cond, num_classes=n_classes).long()
        true_noise = torch.randn_like(x)# noise at x_t
        x_t = diffusion.q_sample(x,t,true_noise).float()

        noise_pred = noise_predictor(x_t, t, cond)
        noise_pred = noise_pred.reshape(B, -1)
        true_noise = true_noise.reshape(B, -1)
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
        os.makedirs('model/ldm', exist_ok=True)
        torch.save(diffusion.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")

    # Evaluate on test set every TEST_INTERVAL epochs
    if (epoch + 1) % TEST_INTERVAL == 0:
        sample()

