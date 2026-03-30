# -*- coding: utf-8 -*-
"""
Video branch for multimodal guitar transcription.

Uses ConvNeXt V2 Atto for frame encoding + TCN for temporal dependencies.

Input: [B, 7, 3, 224, 224] - 7 video frames (280ms @ 25fps)
Output: [B, 512] - video features

Architecture:
- ConvNeXt V2 Atto (pretrained) → [B, 7, 400]
- TCN (3 layers, dilation 1,2,4) → [B, 400]
- Projection → [B, 512]
"""

import torch
import torch.nn as nn

# Handle import when timm not installed
try:
    from timm import create_model
except ImportError:
    create_model = None


class TemporalEncoder(nn.Module):
    """
    Temporal Convolutional Network for video features.
    
    Receptive field: 7 frames (280ms @ 25fps)
    """
    
    def __init__(self, dim: int = 400, dropout: float = 0.2):
        super().__init__()
        
        self.tcn = nn.Sequential(
            # Block 1: RF = 3 frames (120ms)
            nn.Conv1d(dim, dim, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            # Block 2: RF = 5 frames (200ms)
            nn.Conv1d(dim, dim, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            # Block 3: RF = 7 frames (280ms) ✓
            nn.Conv1d(dim, dim, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, dim] - temporal sequence
        
        Returns:
            [B, dim] - temporally pooled features
        """
        # Transpose for Conv1d: [B, T, dim] → [B, dim, T]
        x = x.transpose(1, 2)
        
        # TCN
        x = self.tcn(x)  # [B, dim, T]
        
        # Global temporal pooling
        x = x.mean(dim=-1)  # [B, dim]
        
        return x


class VideoBranch(nn.Module):
    """
    Video branch with ConvNeXt V2 Atto + TCN.
    """
    
    def __init__(self, pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        
        # Check if timm is installed
        if create_model is None:
            raise ImportError(
                "timm is required for VideoBranch. Install with: pip install timm"
            )
        
        # ConvNeXt V2 Atto (smallest, fastest)
        # Output: [B, 320, 7, 7] for 224x224 input
        self.backbone = create_model(
            'convnext_atto',
            pretrained=pretrained,
            num_classes=0,
            global_pool=''  # Keep spatial dimensions
        )
        
        # Get actual feature dimension
        self.feature_dim = 320  # ConvNeXt Atto outputs 320-dim features
        
        # Global pooling to get [B, 320]
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Freeze backbone if needed
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Temporal encoder
        self.temporal_encoder = TemporalEncoder(dim=self.feature_dim, dropout=0.2)
        
        # Projection to match audio branch
        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, 3, 224, 224] - video frames (T=7)
        
        Returns:
            [B, 512] - video features
        """
        B, T, C, H, W = x.shape
        
        # Encode each frame
        # Reshape: [B, T, C, H, W] → [B*T, C, H, W]
        frames = x.view(B * T, C, H, W)
        
        # ConvNeXt: [B*T, C, H, W] → [B*T, 320, 7, 7]
        frame_features = self.backbone(frames)
        
        # Global pool: [B*T, 320, 7, 7] → [B*T, 320]
        frame_features = self.global_pool(frame_features)  # [B*T, 320, 1, 1]
        frame_features = frame_features.view(B * T, self.feature_dim)  # [B*T, 320]
        
        # Reshape: [B*T, 320] → [B, T, 320]
        frame_features = frame_features.view(B, T, self.feature_dim)
        
        # Temporal encoding
        video_features = self.temporal_encoder(frame_features)  # [B, 320]
        
        # Projection
        video_features = self.projection(video_features)  # [B, 512]
        
        return video_features
    
    def freeze_backbone(self):
        """Freeze ConvNeXt backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze ConvNeXt backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = True


def test_model():
    """Test the model with dummy input."""
    model = VideoBranch(pretrained=False)
    
    # Dummy input: batch=2, 7 frames, 3 channels, 224x224
    x = torch.randn(2, 7, 3, 224, 224)
    
    features = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {features.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test freeze/unfreeze
    model.freeze_backbone()
    print(f"\nBackbone frozen: {not any(p.requires_grad for p in model.backbone.parameters())}")
    
    model.unfreeze_backbone()
    print(f"Backbone unfrozen: {any(p.requires_grad for p in model.backbone.parameters())}")


if __name__ == '__main__':
    test_model()
