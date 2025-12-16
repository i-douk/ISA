# Minimal online GMM for streaming data in PyTorch
# Stauffer–Grimson–style: hard assignment, exponential updates

import torch

class OnlineGMM(torch.nn.Module):
    def __init__(self, K=3, lr=None, init_var=30.0, match_thresh=2.5, device="cpu"):
        super().__init__()
        self.K = K
        self.lr = lr
        self.match_thresh = match_thresh
        self.device = device

        # Parameters per component
        self.mu = torch.zeros(K, device=device)
        self.var = torch.ones(K, device=device) * init_var
        # mixing coeff have been initialised in the class defintion as 1/3 for uniformity but will be assigned the 0.05 value as instructed in the tasks sheet
        self.w = torch.ones(K, device=device) / K

    @torch.no_grad()
    def step(self, x):
        """Update model with one observation x (scalar)."""
        x = torch.as_tensor(x, device=self.device, dtype=torch.float32)

        # Mahalanobis distance for matching
        dist = torch.abs(x - self.mu) / torch.sqrt(self.var)
        matches = dist < self.match_thresh

        # Weight decay
        self.w *= (1.0 - self.lr)

        if matches.any():
            k = torch.argmin(dist)
            self.w[k] += self.lr

            # Exponential updates (hard EM)
            rho = self.lr
            diff = x - self.mu[k]
            self.mu[k] += rho * diff
            self.var[k] += rho * (diff * diff - self.var[k])

        # Normalize weights
        self.w /= self.w.sum()

    def score_background(self, x):
        """Likelihood score (higher = more background-like)."""
        x = torch.as_tensor(x, device=self.device, dtype=torch.float32)
        probs = torch.exp(-0.5 * (x - self.mu) ** 2 / self.var) / torch.sqrt(self.var)
        return torch.sum(self.w * probs)

