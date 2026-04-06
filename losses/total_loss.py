import torch
import torch.nn as nn
import torch.nn.functional as F

class LNCC(nn.Module):
    """Local Normalized Cross-Correlation Loss"""
    def __init__(self, window_size=9):
        super().__init__()
        self.window_size = window_size
        self.pad = window_size // 2
    
    def forward(self, I_fixed, I_moved):
        """
        Args:
            I_fixed, I_moved: [B, 1, D, H, W]
        Returns:
            loss: scalar
        """
        # Compute local means
        I_fixed_sum = F.conv3d(I_fixed, torch.ones(1, 1, self.window_size, self.window_size, self.window_size).to(I_fixed.device), padding=self.pad)
        I_moved_sum = F.conv3d(I_moved, torch.ones(1, 1, self.window_size, self.window_size, self.window_size).to(I_moved.device), padding=self.pad)
        
        I_fixed_mean = I_fixed_sum / (self.window_size ** 3)
        I_moved_mean = I_moved_sum / (self.window_size ** 3)
        
        # Compute local variance and covariance
        I_fixed_sq = F.conv3d(I_fixed ** 2, torch.ones(1, 1, self.window_size, self.window_size, self.window_size).to(I_fixed.device), padding=self.pad)
        I_moved_sq = F.conv3d(I_moved ** 2, torch.ones(1, 1, self.window_size, self.window_size, self.window_size).to(I_moved.device), padding=self.pad)
        I_fixed_moved = F.conv3d(I_fixed * I_moved, torch.ones(1, 1, self.window_size, self.window_size, self.window_size).to(I_fixed.device), padding=self.pad)
        
        var_fixed = I_fixed_sq - I_fixed_mean ** 2
        var_moved = I_moved_sq - I_moved_mean ** 2
        cov = I_fixed_moved - I_fixed_mean * I_moved_mean
        
        # LNCC
        cc = cov ** 2 / (var_fixed * var_moved + 1e-8)
        loss = -torch.mean(cc)
        
        return loss

class GradientLoss(nn.Module):
    """Gradient-based regularization loss"""
    def __init__(self):
        super().__init__()
    
    def forward(self, flow):
        """
        Args:
            flow: [B, 3, D, H, W]
        Returns:
            loss: scalar
        """
        dy = torch.abs(flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :])
        dx = torch.abs(flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :])
        dz = torch.abs(flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1])
        
        loss = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        return loss

class TotalLoss(nn.Module):
    """Combined similarity and regularization loss"""
    def __init__(self, lambda_reg=1.0):
        super().__init__()
        self.lambda_reg = lambda_reg
        self.similarity_loss = LNCC()
        self.regularization_loss = GradientLoss()
    
    def forward(self, I_fixed, I_moved, flow):
        """
        Args:
            I_fixed: Fixed image [B, 1, D, H, W]
            I_moved: Warped moving image [B, 1, D, H, W]
            flow: Deformation field [B, 3, D, H, W]
        Returns:
            total_loss, similarity_loss, regularization_loss
        """
        L_sim = self.similarity_loss(I_fixed, I_moved)
        L_reg = self.regularization_loss(flow)
        L_total = L_sim + self.lambda_reg * L_reg
        
        return L_total, L_sim, L_reg