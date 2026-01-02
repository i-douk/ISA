"""
ChatGPT (OpenAI) was used for assistance with boilerplate code,
syntax corrections and suggestions, formatting, and general implementation inquiry.
"""

# Online GMM for background subtraction
# Implements Stauffer & Grimson adaptive Gaussian mixture model per pixel
# PyTorch used for clean tensor ops, personal familiarity and when compatiblefor  switching to GPU
# OpenCV avoided to show full algorithm and hidden logic

import os 
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as transforms
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"

# initializes the GMM using 3D tensors for mean, variance, and mixing coefficients
# will be stored as [H: nb of pixel rows, W: nb of pixel columns, K: k-th Gaussian] per frame
# using tensor ops updates all pixels simultaneously w/o python loops which would run slowly
# Gaussian updates and initialization still occurs per-pixel
class OnlineGMM:
    def __init__(self, H, W, K=3, lr=1/40, init_var=30.0, match_thresh=2.5, device=device):
        self.H, self.W, self.K = H, W, K
        self.lr, self.init_var, self.match_thresh, self.device = lr, init_var, match_thresh, device
        self.mu = torch.zeros(H, W, K, device=device)
        self.var = torch.ones(H, W, K, device=device)*init_var
        self.pi = torch.zeros(H, W, K, device=device)
        self.pi[...,0] = 1.0  # first Gaussian is initialized as dominant

    # Update the Gaussian mixture for every pixel based on new frame
    def update(self, frame):
        # unsqueeze adds a dimension [H,K,1] so the frame can automatically expand to match [H,W,K]
        frame_exp = frame.unsqueeze(-1)
        # distance from pixel to Gaussian, scaled by variance
        dist = torch.abs(frame_exp - self.mu)/torch.sqrt(self.var)
        # Stauffer and Grimson update rule with delta = 0
        self.pi *= (1.0 - self.lr)

        # finds the Gaussian with minimum distance for each pixel
        min_dist, best_match = dist.min(dim=-1)
        # mask for pixels that satisfy the distance threshold
        match_mask = min_dist < self.match_thresh
        # sets unmatched pixels to -1
        best_match = torch.where(match_mask, best_match, torch.full_like(best_match, -1))

        # update matched Gaussians
        for k in range(self.K):
            mask = best_match==k  # pixels matched with k-th Gaussian
            if mask.any():
                rows, cols = torch.nonzero(mask, as_tuple=True)  # matched pixel indices
                diff = frame[rows,cols]-self.mu[rows,cols,k]  # difference [num_matches]
                self.mu[rows,cols,k] += self.lr*diff  # update mean
                self.var[rows,cols,k] += self.lr*(diff**2 - self.var[rows,cols,k])  # update variance
                self.pi[rows,cols,k] += self.lr  # update mixing coefficient

        # initialize unmatched pixels with current frame value
        unmatched = best_match==-1
        if unmatched.any():
            rows, cols = torch.nonzero(unmatched, as_tuple=True)  # unmatched indices
            score = self.pi/self.var  # score to pick weakest Gaussian [H,W,K]
            weakest = score.argmin(dim=-1)  # weakest Gaussian index per pixel
            k = weakest[rows,cols]  # Gaussian to replace
            self.mu[rows,cols,k] = frame[rows,cols]  # new mean = pixel value
            self.var[rows,cols,k] = self.init_var  # large variance
            self.pi[rows,cols,k] = torch.maximum(self.pi[rows,cols,k],
                                                 torch.tensor(0.05, device=self.pi.device))  # soft init pi instead of hard assignement

        self.pi /= self.pi.sum(dim=-1, keepdim=True)  # normalize pi
        score = self.pi/self.var  # sort score
        sorted_idx = score.argsort(dim=-1, descending=True)  # sort Gaussians
        rows = torch.arange(self.H, device=self.mu.device)[:,None,None]
        cols = torch.arange(self.W, device=self.mu.device)[None,:,None]
        self.mu = self.mu[rows,cols,sorted_idx]  # reorder mean
        self.var = self.var[rows,cols,sorted_idx]  # reorder variance
        self.pi = self.pi[rows,cols,sorted_idx]  # reorder mixing coeff

    # get background mean = first Gaussian
    def get_background_mean(self):
        return self.mu[...,0]

    # get background variance using cumulative pi threshold
    def get_background_var(self, T=0.7):
        cum_pi = torch.cumsum(self.pi, dim=-1)  # cumulative sum along K
        bg_mask = cum_pi<=T  # mask of background Gaussians
        bg_mask[...,0] = True  # always include first Gaussian
        bg_var = (self.var*bg_mask).sum(dim=-1)/bg_mask.sum(dim=-1).clamp(min=1)  # weighted var
        return bg_var

    # get foreground mask: pixels not matching background Gaussians
    def get_foreground_mask(self, frame, T=0.7):
        cum_pi = torch.cumsum(self.pi, dim=-1)
        bg_mask = cum_pi<=T
        bg_mask[...,0] = True
        frame_exp = frame.unsqueeze(-1)
        diff = torch.abs(frame_exp - self.mu)
        std = torch.sqrt(self.var)
        match = (diff<=self.match_thresh*std) & bg_mask  # matches background Gaussians
        fg_mask = (~match.any(dim=-1)).to(torch.uint8)  # foreground pixels
        return fg_mask 

# morphological operations for simple denoising
def dilate(mask, kernel_size=3):
    pad = kernel_size//2
    return F.max_pool2d(mask.float(), kernel_size=kernel_size, stride=1, padding=pad)

def erode(mask, kernel_size=3):
    pad = kernel_size//2
    return 1 - F.max_pool2d(1-mask.float(), kernel_size=kernel_size, stride=1, padding=pad)

def opening(mask, kernel_size=3):
    return dilate(erode(mask, kernel_size), kernel_size)

def closing(mask, kernel_size=3):
    return erode(dilate(mask, kernel_size), kernel_size)

def denoise_masks(masks, kernel_size=3):
    masks = masks.unsqueeze(1)
    masks = opening(masks, kernel_size)
    masks = closing(masks, kernel_size)
    return masks.squeeze(1)

# load grayscale frames as [N,C,H,W], scale 0-255
def load_images_as_grayscale(folder_path, device=device):
    image_files = sorted(f for f in os.listdir(folder_path) if f.lower().endswith(".jpeg"))
    frames = []
    # store each image as a tensor and scale to 0-255 because method normalizes pixel values to [0,1]
    for f in image_files:
        img = Image.open(os.path.join(folder_path,f)).convert("L")
        tensor = transforms.to_tensor(img)*255.0
        frames.append(tensor)
    frames = torch.stack(frames, dim=0).to(device)
    print(f"Loaded {len(frames)} frames, shape {frames.shape}")
    return frames, image_files

# save side-by-side FG and BG masks for all frames and learning rates
def save_fg_bg_masks_all(frames_tensor, frame_files, learning_rates=[1/40,1/1400], output_folder="all_masks"):
    if frames_tensor.ndim == 4:
        frames_tensor = frames_tensor.squeeze(1)
    N, H, W = frames_tensor.shape

    for lr in learning_rates:
        print(f"\nProcessing all frames with lr={lr}...")
        gmm = OnlineGMM(H, W, K=3, lr=lr, device=device)
        lr_folder = os.path.join(output_folder, f"lr_{lr}")
        os.makedirs(lr_folder, exist_ok=True)

        for t in range(N):
            frame = frames_tensor[t]
            gmm.update(frame)

            fg_mask = gmm.get_foreground_mask(frame)
            fg_mask_denoised = denoise_masks(fg_mask)
            bg_mask = 1 - fg_mask_denoised

            combined = torch.cat([fg_mask_denoised, bg_mask], dim=1)

            plt.figure(figsize=(10,5))
            plt.imshow(combined.cpu(), cmap="gray")
            plt.axis('off')
            plt.title(f"{frame_files[t]} | lr={lr}")
            plt.tight_layout()
            plt.savefig(os.path.join(lr_folder, f"frame_{t}_{frame_files[t]}"))
            plt.close()

        print(f"Finished learning rate {lr}")

# process selected frames and save plots for multiple learning rates
def process_sequence_lr(frames_tensor, frame_files, frame_indices_to_show, learning_rates):
    N,C,H,W = frames_tensor.shape
    frames_tensor = frames_tensor.squeeze(1)
    for lr in learning_rates:
        print(f"\nProcessing with lr={lr}")
        gmm = OnlineGMM(H,W,K=3,lr=lr,device=device)
        for t in range(N):
            frame = frames_tensor[t]
            gmm.update(frame)

            fg_mask = gmm.get_foreground_mask(frame)
            fg_mask_denoised = denoise_masks(fg_mask)

            if t in frame_indices_to_show:
                bg_mean = gmm.get_background_mean() 
                bg_var = gmm.get_background_var()
                diff = torch.abs(frame-bg_mean)
                bg_mask = 1 - fg_mask_denoised

                os.makedirs(f"results_lr_{lr}", exist_ok=True)
                plt.figure(figsize=(12,6))
                plt.subplot(2,3,1); plt.imshow(frame.cpu(), cmap='gray'); plt.axis('off'); plt.title("Frame")
                plt.subplot(2,3,2); plt.imshow(bg_mean.cpu(), cmap='gray'); plt.axis('off'); plt.title("BG mean")
                plt.subplot(2,3,3); plt.imshow(bg_var.cpu(), cmap='hot'); plt.axis('off'); plt.title("BG var")
                plt.subplot(2,3,4); plt.imshow(diff.cpu(), cmap='gray'); plt.axis('off'); plt.title("Diff")
                plt.subplot(2,3,5); plt.imshow(fg_mask_denoised.cpu(), cmap='gray'); plt.axis('off'); plt.title("FG mask")
                plt.subplot(2,3,6); plt.imshow(bg_mask.cpu(), cmap='gray'); plt.axis('off'); plt.title("BG mask")
                plt.suptitle(frame_files[t])
                plt.tight_layout()
                plt.savefig(f"results_lr_{lr}/frame_{t}_{frame_files[t]}.png")
                plt.close()
        print(f"Finished lr={lr}")

# main execution
if __name__ == "__main__":
    folder = "sequence"  # folder with JPEG frames
    frames_tensor, frame_files = load_images_as_grayscale(folder)
    frame_indices_to_show = [2422,2794]  # selected frames
    learning_rates = [1/40,1/1400]

    process_sequence_lr(frames_tensor, frame_files, frame_indices_to_show, learning_rates)
    save_fg_bg_masks_all(frames_tensor, frame_files, learning_rates, output_folder="all_masks")
    print("Processing complete")