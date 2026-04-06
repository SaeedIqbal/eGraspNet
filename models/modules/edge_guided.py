import torch
import torch.nn as nn
import torch.nn.functional as F

class SobelOperator3D:
    """3D Sobel operator for edge detection"""
    def __init__(self, device='cuda'):
        self.device = device
        self.kernels = self._create_kernels()
    
    def _create_kernels(self):
        """Create 3D Sobel kernels for X, Y, Z axes"""
        # X-axis kernel
        kx = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           [[-2, 0, 2], [-4, 0, 4], [-2, 0, 2]],
                           [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], 
                          dtype=torch.float32, device=self.device)
        
        # Y-axis kernel
        ky = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           [[-2, -4, -2], [0, 0, 0], [2, 4, 2]],
                           [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], 
                          dtype=torch.float32, device=self.device)
        
        # Z-axis kernel
        kz = torch.tensor([[[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]],
                           [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                           [[1, 2, 1], [2, 4, 2], [1, 2, 1]]], 
                          dtype=torch.float32, device=self.device)
        
        return [kx.unsqueeze(0).unsqueeze(0), 
                ky.unsqueeze(0).unsqueeze(0), 
                kz.unsqueeze(0).unsqueeze(0)]
    
    def compute_gradient(self, image):
        """Compute gradient magnitude using 3D Sobel operator"""
        Gx = F.conv3d(image, self.kernels[0], padding=1)
        Gy = F.conv3d(image, self.kernels[1], padding=1)
        Gz = F.conv3d(image, self.kernels[2], padding=1)
        
        gradient_magnitude = torch.sqrt(Gx**2 + Gy**2 + Gz**2 + 1e-8)
        return gradient_magnitude

class EdgeGuidedModule(nn.Module):
    """Edge-Guided (EG) Module for structural boundary enhancement"""
    def __init__(self, device='cuda'):
        super().__init__()
        self.sobel = SobelOperator3D(device=device)
    
    def forward(self, image):
        """
        Args:
            image: Input image tensor [B, 1, D, H, W]
        Returns:
            augmented_image: [B, 2, D, H, W] with original + edge channels
        """
        edge_map = self.sobel.compute_gradient(image)
        augmented = torch.cat([image, edge_map], dim=1)
        return augmented