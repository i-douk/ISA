# Gaussian Mixture Model (GMM) with online k-means approx. 
# Stauffer & Grimson Algorithm for background subtraction
# Based on original paper 'Adaptive background mixture models for real-time tracking' and lab instructions
# PyTorch was used for familiarity, clear tensor expressions
# OpenCV, although it offers out of the box classes for this assignement is not used intentionally to avoid hiding any logic of the implementation of the Stauffer and Grimson algorithm as well as the morphological denoising step

import os
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as transforms

device = "cpu"

# Definition of the GMM class and initialisations, nn.Module here is used as a container not as a neural network
class OnlineGMM(torch.nn.Module):
    def __init__(self, K=3, lr=1/40, init_var=30.0, match_thresh=2.5, device=device):
        self.K = K
        self.lr = lr
        self.match_thresh = match_thresh
        self.device = device
        self.mu = torch.zeros(K, device=device)
        self.var = torch.ones(K, device=device) * init_var
        self.pi = torch.ones(K, device=device) / K

    # Not a neural network so no learning via gradients
    @torch.no_grad()

    # Func update_gaussian takes single pixel x from a frame, updates the mixture of Gaussians for x depending on matching iwth existing gaussians
    def update_gaussians(self, x):
        # makes x a torch tensor, ensure it lives in the correct device, and define type for type consistency
        x = torch.as_tensor(x, device=self.device, dtype=torch.float32)
        # this calculates 'Mahalanobis distance' to determine if a new pixel value belongs to an existing gaussian 
        dist = torch.abs(x - self.mu) / torch.sqrt(self.var)
        matches = dist < self.match_thresh
        # mixing coeffiscient decay
        self.pi *= (1.0 - self.lr)
        # describes case where match is found, we update our values accordingly
        if matches.any():
            k = torch.argmin(dist)
            self.pi[k] += self.lr
            diff = x - self.mu[k]
            self.mu[k] += self.lr * diff
            self.var[k] += self.lr * (diff * diff - self.var[k])
        #describes case where no match is found, we create new gaussian is introduced around new observation with weight initialised with 0.05
        else:
            k = torch.argmin(self.pi)
            self.mu[k] = x
            self.var[k] = 30.0
            self.pi[k] = 0.05
        # normalized our mixing coeffiscients so that sum(pi) = 1
        self.pi /= self.pi.sum()

    # defines strongest weights as belonging to the background
    def background_index(self):
      score = self.pi / torch.sqrt(self.var)
      return torch.argmax(score).item()
    def is_background(self, x):
      k = self.background_index()
      x = torch.as_tensor(x, device=self.device, dtype=torch.float32)
      return torch.abs(x - self.mu[k]) < self.match_thresh * torch.sqrt(self.var[k])

# loads images and converts to grayscale
def load_images_as_grayscale(folder_path, device=device):
    image_files = sorted(
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".jpeg")) # extend file estensions when needed
    )

    frames = []
    for f in image_files:
        # converts our images to grayscale i.e. one channel
        img = Image.open(os.path.join(folder_path, f)).convert("L")
        tensor = transforms.to_tensor(img)
        frames.append(tensor)

    # put all images in one tensor (N, H, W) , N is the number of images
    frames = torch.stack(frames, dim=0) 
    print(f" Shape of the tensor for frames after loading images as grayscale: {frames.shape}")

    return frames


# initializes models per pixel, i.e. creates a mixture of 3 gaussians per pixel
def initialize_pixel_models(H, W, K=3, lr=1/40):
    return [[OnlineGMM(K=K, lr=lr, device=device) for _ in range(W)] for _ in range(H)]

# runs GMM on sequence
def run_gmm_sequence(frames, lr):
    frames = frames.squeeze(1) 
    num_frames, H, W = frames.shape
    # initialize a separate OnlineGMM per pixel (H x W grid)
    models = initialize_pixel_models(H, W, K=3, lr=lr)
    foreground_masks = []  # list to store masks for all frames

    # iterate over each frame in the sequence
    for t in range(num_frames):
        frame = frames[t].cpu()
        # initializes a foreground mask for this frame
        # dtype=uint8 because mask will contain 0 (background) or 1 (foreground)
        fg_mask = torch.zeros((H, W), device=device, dtype=torch.uint8)

        # iterate over each pixel
        for i in range(H):
            for j in range(W):
                model = models[i][j]  # get the GMM for this pixel
                model.lr = lr  # update learning rate for this frame
                # update Gaussian mixture for the pixel based on its new value
                model.update_gaussians(frame[i, j])
                # if pixel does not match background Gaussian, mark as foreground
                if not model.is_background(frame[i, j]):
                    fg_mask[i, j] = 1

        # append mask to list (move to CPU for storage)
        foreground_masks.append(fg_mask.cpu())

    # stack all masks into a single tensor of shape (num_frames, H, W)
    return torch.stack(foreground_masks, dim=0)

# Morphological denoising in pure torch
def dilate(mask, kernel_size=3):
    pad = kernel_size // 2
    return F.max_pool2d(mask.float(), kernel_size=kernel_size, stride=1, padding=pad)

def erode(mask, kernel_size=3):
    pad = kernel_size // 2
    return 1 - F.max_pool2d((1 - mask.float()), kernel_size=kernel_size, stride=1, padding=pad)

def opening(mask, kernel_size=3):
    return dilate(erode(mask, kernel_size), kernel_size)

def closing(mask, kernel_size=3):
    return erode(dilate(mask, kernel_size), kernel_size)

def denoise_masks(masks, kernel_size=3):
    masks = masks.unsqueeze(1)
    masks = opening(masks, kernel_size)
    masks = closing(masks, kernel_size)
    return masks.squeeze(1)

# starts evaluation evaluation
if __name__ == "__main__":
    folder = "sequence"
    frames_tensor = load_images_as_grayscale(folder)

    learning_rates = [1/40, 1/1400]
    results = {}

    for lr in learning_rates:
        print(f"Processing sequence with learning rate {lr}...")
        fg_masks = run_gmm_sequence(frames_tensor, lr)
        fg_masks_denoised = denoise_masks(fg_masks)
        results[lr] = fg_masks_denoised
        print(f"Done learning rate {lr}, masks shape: {fg_masks.shape}")

    print("All processing complete.")