import numpy as np
import torch

def compute_dice(fixed_seg, moved_seg):
    """
    Compute Dice Similarity Coefficient
    Args:
        fixed_seg, moved_seg: Binary segmentation masks
    Returns:
        dice: float
    """
    intersection = np.sum(fixed_seg * moved_seg)
    union = np.sum(fixed_seg) + np.sum(moved_seg)
    
    if union == 0:
        return 1.0
    
    dice = 2.0 * intersection / union
    return dice

def compute_jacobian_determinant(flow):
    """
    Compute Jacobian determinant of deformation field
    Args:
        flow: [B, 3, D, H, W] deformation field
    Returns:
        jacobian: [B, D, H, W]
    """
    device = flow.device
    B, _, D, H, W = flow.shape
    
    # Compute spatial gradients
    dx = flow[:, 0, :, :, :]
    dy = flow[:, 1, :, :, :]
    dz = flow[:, 2, :, :, :]
    
    # Gradient computation
    grad_x = torch.zeros_like(flow)
    grad_y = torch.zeros_like(flow)
    grad_z = torch.zeros_like(flow)
    
    grad_x[:, 0, 1:, :, :] = (dx[:, 1:, :, :] - dx[:, :-1, :, :])
    grad_y[:, 1, :, 1:, :] = (dy[:, :, 1:, :] - dy[:, :, :-1, :])
    grad_z[:, 2, :, :, 1:] = (dz[:, :, :, 1:] - dz[:, :, :, :-1])
    
    # Jacobian matrix
    J11 = 1 + grad_x[:, 0, :, :, :]
    J12 = grad_x[:, 1, :, :, :]
    J13 = grad_x[:, 2, :, :, :]
    J21 = grad_y[:, 0, :, :, :]
    J22 = 1 + grad_y[:, 1, :, :, :]
    J23 = grad_y[:, 2, :, :, :]
    J31 = grad_z[:, 0, :, :, :]
    J32 = grad_z[:, 1, :, :, :]
    J33 = 1 + grad_z[:, 2, :, :, :]
    
    # Determinant
    jacobian = (J11 * (J22 * J33 - J23 * J32) -
                J12 * (J21 * J33 - J23 * J31) +
                J13 * (J21 * J32 - J22 * J31))
    
    return jacobian

def compute_folding_percentage(jacobian):
    """
    Compute percentage of folding voxels (non-positive Jacobian)
    Args:
        jacobian: [B, D, H, W]
    Returns:
        folding_percentage: float
    """
    folding_voxels = torch.sum(jacobian <= 0)
    total_voxels = jacobian.numel()
    percentage = (folding_voxels / total_voxels) * 100.0
    return percentage.item()