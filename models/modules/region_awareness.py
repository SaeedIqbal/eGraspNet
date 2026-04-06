import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CAMGenerator(nn.Module):
    """Class Activation Map (CAM) Generator"""
    def __init__(self, in_channels):
        super().__init__()
        self.conv1x1 = nn.Conv3d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, features):
        """
        Args:
            features: [B, C, D, H, W]
        Returns:
            cam: [B, 1, D, H, W]
        """
        cam = self.conv1x1(features)
        cam = self.sigmoid(cam)
        return cam

class PatchCAMAggregator(nn.Module):
    """Aggregate voxel-level CAM to patch-level saliency"""
    def __init__(self, patch_size=(4, 4, 4)):
        super().__init__()
        self.patch_size = patch_size
        self.pool = nn.AvgPool3d(kernel_size=patch_size, stride=patch_size)
    
    def forward(self, cam):
        """
        Args:
            cam: [B, 1, D, H, W]
        Returns:
            patch_cam: [B, N, 1] where N = number of patches
        """
        patch_cam = self.pool(cam)
        B, _, D, H, W = patch_cam.shape
        patch_cam = patch_cam.view(B, 1, -1).transpose(1, 2)
        return patch_cam

class MutualCrossAttention(nn.Module):
    """Mutual Cross Attention with CAM modulation"""
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.W_Q = nn.Linear(embed_dim, embed_dim)
        self.W_K = nn.Linear(embed_dim, embed_dim)
        self.W_V = nn.Linear(embed_dim, embed_dim)
        self.W_O = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, T_m, T_f, cam_m, cam_f):
        """
        Args:
            T_m, T_f: Token embeddings [B, N, C]
            cam_m, cam_f: Patch-level CAM [B, N, 1]
        Returns:
            T_hat_m, T_hat_f: Enhanced tokens [B, N, C]
        """
        # Linear projections
        Q_m = self.W_Q(T_m)
        K_f = self.W_K(T_f)
        V_f = self.W_V(T_f)
        
        Q_f = self.W_Q(T_f)
        K_m = self.W_K(T_m)
        V_m = self.W_V(T_m)
        
        # Scaled dot-product attention
        A_m2f = torch.softmax((Q_m @ K_f.transpose(-2, -1)) / self.scale, dim=-1)
        A_f2m = torch.softmax((Q_f @ K_m.transpose(-2, -1)) / self.scale, dim=-1)
        
        # CAM modulation
        cam_matrix_m2f = cam_m @ cam_f.transpose(-2, -1)
        cam_matrix_f2m = cam_f @ cam_m.transpose(-2, -1)
        
        A_m2f_prime = A_m2f * cam_matrix_m2f
        A_f2m_prime = A_f2m * cam_matrix_f2m
        
        # Aggregate values
        T_hat_m = A_m2f_prime @ V_f
        T_hat_f = A_f2m_prime @ V_m
        
        # Output projection
        T_hat_m = self.W_O(T_hat_m)
        T_hat_f = self.W_O(T_hat_f)
        
        return T_hat_m, T_hat_f

class RegionAwarenessModule(nn.Module):
    """Complete Region Awareness (RA) Module"""
    def __init__(self, in_channels, embed_dim, num_heads=8, patch_size=(4, 4, 4)):
        super().__init__()
        self.cam_generator = CAMGenerator(in_channels)
        self.patch_aggregator = PatchCAMAggregator(patch_size)
        self.attention = MutualCrossAttention(embed_dim, num_heads)
        self.patch_size = patch_size
    
    def forward(self, E_M, E_F):
        """
        Args:
            E_M, E_F: Feature maps [B, C, D, H, W]
        Returns:
            E_hat_M, E_hat_F: Enhanced features [B, C, D, H, W]
        """
        # Generate CAM
        cam_M = self.cam_generator(E_M)
        cam_F = self.cam_generator(E_F)
        
        # Aggregate to patch level
        cam_M_patch = self.patch_aggregator(cam_M)
        cam_F_patch = self.patch_aggregator(cam_F)
        
        # Tokenize features
        B, C, D, H, W = E_M.shape
        pD, pH, pW = self.patch_size
        
        T_m = E_M.view(B, C, D//pD, pD, H//pH, pH, W//pW, pW)
        T_m = T_m.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
        T_m = T_m.view(B, -1, C * pD * pH * pW)
        
        T_f = E_F.view(B, C, D//pD, pD, H//pH, pH, W//pW, pW)
        T_f = T_f.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
        T_f = T_f.view(B, -1, C * pD * pH * pW)
        
        # Apply mutual cross attention
        T_hat_m, T_hat_f = self.attention(T_m, T_f, cam_M_patch, cam_F_patch)
        
        # Reshape back to volumetric space
        # (Implementation depends on specific architecture)
        
        return E_M, E_F  # Placeholder