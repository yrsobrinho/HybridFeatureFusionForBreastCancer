import torch
from typing import List, Tuple

class CosineAnnealingAlpha:
    def __init__(self, alpha_max, alpha_min, T_max, warmup_epochs=0):
        self.alpha_max = alpha_max
        self.alpha_min = alpha_min
        self.total_epochs = T_max
        self.warmup_epochs = warmup_epochs
        self.T_max = T_max - warmup_epochs  
        
    def get_alpha(self, epoch):
        if epoch < self.warmup_epochs:
            return 0.0 
        
        adjusted_epoch = epoch - self.warmup_epochs
        
        progress = adjusted_epoch / self.T_max  # 0 to 1
        cos_value = torch.cos(torch.tensor(progress * torch.pi))
        alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * (1 + cos_value) / 2
        
        return alpha.item()
    
    def plot_schedule(self, total_epochs):
        """Visualize the alpha schedule"""
        epochs = list(range(total_epochs))
        alphas = [self.get_alpha(e) for e in epochs]
        
        print("Alpha Schedule:")
        print(f"Warmup epochs: {self.warmup_epochs}")
        print(f"Cosine epochs: {self.T_max}")
        print(f"Alpha range: {self.alpha_min:.3f} → {self.alpha_max:.3f} → {self.alpha_min:.3f}")
        
        for i in range(0, total_epochs, 2):  # Every 2 epochs
            status = "WARMUP" if i < self.warmup_epochs else "COSINE"
            print(f"Epoch {i+1:2d}: α = {alphas[i]:.3f} [{status}]")
        
        return epochs, alphas
