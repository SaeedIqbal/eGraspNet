import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialTransformer(nn.Module):
    """Spatial Transformer Network for warping"""
    def __init__(self, size, mode='bilinear'):
        super().__init__()
        self.size = size
        self.mode = mode
    
    def forward(self, source, flow):
        """
        Args:
            source: [B, C, D, H, W]
            flow: [B, 3, D, H, W] displacement field
        Returns:
            warped: [B, C, D, H, W]
        """
        device = flow.device
        B, _, D, H, W = flow.shape
        
        # Create sampling grid
        vectors = [torch.arange(0, s) for s in (D, H, W)]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids, dim=0).unsqueeze(0).float().to(device)
        
        # Add displacement
        grid = grid + flow
        
        # Normalize to [-1, 1]
        grid[:, 0] = 2.0 * grid[:, 0] / max(D - 1, 1) - 1.0
        grid[:, 1] = 2.0 * grid[:, 1] / max(H - 1, 1) - 1.0
        grid[:, 2] = 2.0 * grid[:, 2] / max(W - 1, 1) - 1.0
        
        grid = grid.permute(0, 2, 3, 4, 1)  # [B, D, H, W, 3]
        
        # Sample
        warped = F.grid_sample(source, grid, mode=self.mode, align_corners=True)
        return warped

class DeformationFieldEstimator(nn.Module):
    """Predict deformation field from features"""
    def __init__(self, in_channels, out_channels=3):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
    
    def forward(self, features):
        """
        Args:
            features: [B, C, D, H, W]
        Returns:
            flow: [B, 3, D, H, W]
        """
        flow = self.conv(features)
        return flow

class SemiPyramidNetwork(nn.Module):
    """Semi-Pyramid Network for hierarchical deformation estimation"""
    def __init__(self, decoder_channels, start_layer=3):
        super().__init__()
        self.start_layer = start_layer
        self.num_layers = len(decoder_channels)
        
        # Deformation estimators for layers 3, 2, 1
        self.dfe_3 = DeformationFieldEstimator(decoder_channels[2])
        self.dfe_2 = DeformationFieldEstimator(decoder_channels[1])
        self.dfe_1 = DeformationFieldEstimator(decoder_channels[0])
        
        # Spatial transformers
        self.stn = SpatialTransformer(None)
    
    def compose_fields(self, phi1, phi2):
        """
        Compose two deformation fields
        phi_total(x) = phi1(x) + phi2(x + phi1(x))
        """
        phi2_warped = self.stn(phi2, phi1)
        phi_total = phi1 + phi2_warped
        return phi_total
    
    def forward(self, decoder_features):
        """
        Args:
            decoder_features: List of feature maps from decoder [F1, F2, F3, ...]
        Returns:
            phi_final: Final deformation field [B, 3, D, H, W]
        """
        # Start from layer 3 (index 2)
        phi_3 = self.dfe_3(decoder_features[2])
        phi_cumulative = phi_3
        
        # Layer 2
        phi_3_up = F.interpolate(phi_3, scale_factor=2, mode='trilinear', align_corners=True)
        D_M_2_warped = self.stn(decoder_features[1][:, :decoder_features[1].shape[1]//2], phi_3_up)
        D_F_2_warped = self.stn(decoder_features[1][:, decoder_features[1].shape[1]//2:], -phi_3_up)
        features_2 = torch.cat([D_M_2_warped, D_F_2_warped], dim=1)
        
        phi_2 = self.dfe_2(features_2)
        phi_cumulative = self.compose_fields(phi_cumulative, phi_2)
        
        # Layer 1
        phi_cumulative_up = F.interpolate(phi_cumulative, scale_factor=2, mode='trilinear', align_corners=True)
        D_M_1_warped = self.stn(decoder_features[0][:, :decoder_features[0].shape[1]//2], phi_cumulative_up)
        D_F_1_warped = self.stn(decoder_features[0][:, decoder_features[0].shape[1]//2:], -phi_cumulative_up)
        features_1 = torch.cat([D_M_1_warped, D_F_1_warped], dim=1)
        
        phi_1 = self.dfe_1(features_1)
        phi_final = self.compose_fields(phi_cumulative, phi_1)
        
        return phi_final