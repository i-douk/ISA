# Online Gaussian Mixture Model (GMM) for background subtraction
# Implementation of the Stauffer & Grimson algorithm (Adaptive background mixture models) for background subtraction
# Use of PyTorch for clear tensor expressions, familiarity with the framework, and esy switching to GPU
# OpenCV intentionaly avoided to showcase the full algorithm and avoid built-in and obstructed logic

import os
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as transforms
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

device = "cpu"

# initializes the GMM using 3D tensors for mean, variance, and mixing coefficients
# will be stored as [H: nb of pixel rows, W: nb of pixel columns, K: k-th Gaussian] per frame
# using tensor ops updates all pixels simultaneously w/o python loops which would run slowly
# Gaussian updates and initialization still occurs per-pixel

class OnlineGMM:
    def __init__(self, H, W, K=3, lr=1/40, init_var=30.0, match_thresh=2.5, device=device):
        self.H = H
        self.W = W
        self.K = K
        self.lr = lr
        self.match_thresh = match_thresh
        self.device = device
        self.mu = torch.zeros(H, W, K, device=device)
        self.var = torch.ones(H, W, K, device=device) * init_var
        self.pi = torch.ones(H, W, K, device=device) / K

    # Update the Gaussian mixture for every pixel based on new frame
    def update(self, frame):
        # unsqueeze adds a dimension [H,K,1] so the frame can automatically expand to match [H,W,K]
        frame_exp = frame.unsqueeze(-1)
        # distance from pixel to Gaussian, scaled by variance
        dist = torch.abs(frame_exp - self.mu) / torch.sqrt(self.var)
        # Stauffer and Grimson update rule with delta = 0
        self.pi *= (1.0 - self.lr)
        # finds the Gaussian with minimum distance for each pixel
        min_dist, best_match = dist.min(dim=-1)
        # mask for pixels that satisfy the distance threshold
        match_mask = min_dist < self.match_thresh
        # sets unmatched pixels to -1
        best_match[~match_mask] = -1

        # for each Gaussian, we check whether it is the best match for the given frame
        for k in range(self.K):
            # boolean mask indicating which pixels are best matched to Gaussian k
            mask = best_match == k
            # proceed only if at least one pixel matched this Gaussian (update rule)
            if mask.any():
                # retrieves the (row, column) indices of pixels matched to Gaussian k
                rows, cols = torch.nonzero(mask, as_tuple=True)
                # computes the difference between current pixel values and the Gaussian mean
                diff = frame[rows, cols] - self.mu[rows, cols, k]
                # updates the mean to move it toward the current pixel 
                self.mu[rows, cols, k] += self.lr * diff
                # updates the variance according to update rule
                self.var[rows, cols, k] += self.lr * (diff**2 - self.var[rows, cols, k])
                # increases the mixing coefficient (weight) since this Gaussian matched the pixels
                self.pi[rows, cols, k] += self.lr

        # normalizes the mixing coefficients so that sum = 1
        self.pi /= self.pi.sum(dim=-1, keepdim=True)

        score = self.pi / self.var
        sorted_idx = score.argsort(dim=-1, descending=True)  # highest score first
        
        rows = torch.arange(self.H)[:, None, None].expand(self.H, self.W, self.K)
        cols = torch.arange(self.W)[None, :, None].expand(self.H, self.W, self.K)
        
        self.mu  = self.mu[rows, cols, sorted_idx]
        self.var = self.var[rows, cols, sorted_idx]
        self.pi  = self.pi[rows, cols, sorted_idx]
    
    # returns mean value of the background Gaussian for each pixel
    def get_background_mean(self):
        return self.mu[..., 0]
    
    # returns [H, W] tensor of background variance
    def get_background_var(self):
        return self.var[..., 0] 
    
    # returns binary mask for foreground by applying condition on threshold
    def get_foreground_mask(self, frame, T=0.7):
        """
        Returns a [H, W] binary foreground mask using cumulative mixing coefficients.
        Pixels are foreground if they do not match any of the first B background Gaussians.
        
        Parameters:
            frame : [H, W] tensor of the current frame
            T     : threshold for cumulative mixing coefficient to define background
        """
        # sorted Gaussians assumed: mu[...,0] = most likely, mu[...,1], ...
        cum_pi = torch.cumsum(self.pi, dim=-1)  # cumulative sum along K dimension
        
        # mask of background Gaussians per pixel: True if cumulative pi <= T
        bg_mask = cum_pi <= T
        # always include the first Gaussian
        bg_mask[..., 0] = True
        
        # expand frame to [H,W,K] to compare with all Gaussians
        frame_exp = frame.unsqueeze(-1)  # [H,W,1]
        
        # absolute difference to all Gaussians
        diff = torch.abs(frame_exp - self.mu)
        
        # standard deviations
        std = torch.sqrt(self.var)
        
        # pixel matches a background Gaussian if diff < match_thresh * std
        match = diff <= self.match_thresh * std
        
        # only consider Gaussians flagged as background
        match = match & bg_mask
        
        # pixel is foreground if it does NOT match any background Gaussian
        fg_mask = (~match.any(dim=-1)).to(torch.uint8)
        
        return fg_mask

# morphological operations for denoising
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

# loads images as grayscale
def load_images_as_grayscale(folder_path, device=device):
    image_files = sorted(f for f in os.listdir(folder_path) if f.lower().endswith(".jpeg"))
    frames = []
    for f in image_files:
        img = Image.open(os.path.join(folder_path, f)).convert("L")
        # store each image as a tensor and scale to 0-255 because method normalizes pixel values to [0,1]
        tensor = transforms.to_tensor(img) * 255.0
        frames.append(tensor)
    frames = torch.stack(frames, dim=0).to(device)
    print(f"Loaded {len(frames)} frames, shape: {frames.shape}")
    return frames, image_files

#
def process_sequence_lr(frames_tensor, frame_files, frame_indices_to_show, learning_rates):
    N, C, H, W = frames_tensor.shape
    frames_tensor = frames_tensor.squeeze(1)
    for lr in learning_rates:
        print(f"\nProcessing sequence with learning rate {lr}...")
        gmm = OnlineGMM(H, W, K=3, lr=lr, device=device)
        for t in range(N):
            frame = frames_tensor[t]
            gmm.update(frame)
            fg_mask = gmm.get_foreground_mask(frame)
            fg_mask_denoised = denoise_masks(fg_mask)
            if t in frame_indices_to_show:
                bg_mean = gmm.get_background_mean()
                bg_var = gmm.get_background_var()
                diff = torch.abs(frame - bg_mean)
                bg_mask = 1 - fg_mask_denoised
                os.makedirs(f"results_lr_{lr}", exist_ok=True)
                plt.figure(figsize=(12,6))
                plt.subplot(2,3,1)
                plt.imshow(frame.cpu(), cmap='gray'); plt.axis('off'); plt.title("Current Frame")
                plt.subplot(2,3,2)
                plt.imshow(bg_mean.cpu(), cmap='gray'); plt.axis('off'); plt.title("Background Mean")
                plt.subplot(2,3,3)
                plt.imshow(bg_var.cpu(), cmap='hot'); plt.axis('off'); plt.title("Background Variance")
                plt.subplot(2,3,4)
                plt.imshow(diff.cpu(), cmap='gray'); plt.axis('off'); plt.title("Difference")
                plt.subplot(2,3,5)
                plt.imshow(fg_mask_denoised.cpu(), cmap='gray'); plt.axis('off'); plt.title("Foreground Mask")
                plt.subplot(2,3,6)
                plt.imshow(bg_mask.cpu(), cmap='gray'); plt.axis('off'); plt.title("Background Mask")
                plt.suptitle(frame_files[t])
                plt.tight_layout()
                plt.savefig(f"results_lr_{lr}/frame_{t}_{frame_files[t]}.png")
                plt.close()
        print(f"Finished learning rate {lr}")

if __name__ == "__main__":
    folder = "sequence"
    frames_tensor, frame_files = load_images_as_grayscale(folder)
    frame_indices_to_show = [2794, 3019]
    learning_rates = [1/40, 1/1400]
    process_sequence_lr(frames_tensor, frame_files, frame_indices_to_show, learning_rates)
    print("All processing complete")
