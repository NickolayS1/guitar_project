# -*- coding: utf-8 -*-
"""
Dual-Branch CNN model for guitar transcription.

Architecture with:
- Early split after 1 conv layer
- Asymmetric convolutions: (3,1) for onset, (1,7) for pitch
- SE attention in pitch branch
- Residual connections in both branches
- Fusion with 1x1 conv + dropout
- Dual heads for onset and pitch

~106K parameters (optimized for harmonic modeling).

Input: [B, 1, 13, 72] - CQT spectrogram
Output: 
    - onset_pred: [B, 6] - sigmoid probabilities
    - pitch_pred: [B, 6] - normalized MIDI (0-1 range)
"""

import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """
    Channel-wise Squeeze-and-Excitation block.
    
    Compresses spatial dimensions and learns channel-wise attention.
    """
    
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, C, T, F]
        
        Returns:
            Scaled tensor [B, C, T, F]
        """
        B, C, _, _ = x.size()
        
        # Squeeze: global average pooling
        w = self.avg_pool(x).view(B, C)  # [B, C]
        
        # Excitation: FC layers
        w = self.fc(w).view(B, C, 1, 1)  # [B, C, 1, 1]
        
        # Scale: multiply input by attention weights
        return x * w


class DualBranchCNN(nn.Module):
    """
    Dual-Branch CNN with early split and asymmetric convolutions.
    
    Architecture:
    - Shared encoder: 1 conv layer
    - Onset branch: 3x temporal convolutions (3,1)
    - Pitch branch: 4x frequency convolutions (1,7) + SE attention
    - Fusion: 1x1 conv + dropout
    - Dual heads with different pooling strategies
    """
    
    def __init__(
        self,
        n_strings: int = 6,
        encoder_channels: list = None,
        head_hidden: int = 64,
        dropout: float = 0.3,
        se_reduction: int = 8
    ):
        super().__init__()
        
        self.n_strings = n_strings
        
        # Default channels if not specified
        if encoder_channels is None:
            encoder_channels = [32, 64, 96]
        
        c1, c2, c3 = encoder_channels
        
        # === Shared Encoder (1 layer) ===
        # [B, 1, 13, 72] → [B, c1, 13, 72]
        self.shared_encoder = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True)
        )
        
        # ============================================
        # Onset Branch (Temporal features)
        # ============================================
        # 3x Conv(3,1) to capture attack patterns over time
        self.onset_branch = nn.Sequential(
            # Block 1
            nn.Conv2d(c1, c1, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            
            # Block 2
            nn.Conv2d(c1, c1, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            
            # Block 3
            nn.Conv2d(c1, c1, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            
            # Frequency pooling to reduce dimensionality
            nn.MaxPool2d(kernel_size=(1, 2))  # [B, c1, 13, 36]
        )
        
        # Residual connection for onset branch
        self.onset_residual = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=1),  # 1x1 conv for matching dimensions
            nn.BatchNorm2d(c1)
        )
        
        # ============================================
        # Pitch Branch (Frequency features)
        # ============================================
        # 4x Conv(1,7) to capture harmonic relationships
        self.pitch_branch = nn.Sequential(
            # Block 1: Wide kernel for harmonics
            nn.Conv2d(c1, c1, kernel_size=(1, 7), padding=(0, 3)),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            
            # Block 2: Wide kernel for intervals
            nn.Conv2d(c1, c1, kernel_size=(1, 7), padding=(0, 3)),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            
            # Block 3: Medium kernel for chords
            nn.Conv2d(c1, c1, kernel_size=(1, 5), padding=(0, 2)),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            
            # Block 4: Narrow kernel for fine-tuning
            nn.Conv2d(c1, c1, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            
            # SE Attention for channel-wise feature selection
            SEBlock(c1, reduction=se_reduction),
            
            # Frequency pooling
            nn.MaxPool2d(kernel_size=(1, 2))  # [B, c1, 13, 36]
        )
        
        # Residual connection for pitch branch
        self.pitch_residual = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=1),
            nn.BatchNorm2d(c1)
        )
        
        # ============================================
        # Fusion Layer
        # ============================================
        # Concatenate branches: [B, 2*c1, 13, 36]
        # 1x1 conv to mix features and reduce channels
        self.fusion = nn.Sequential(
            nn.Conv2d(c1 * 2, c1, kernel_size=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )
        
        # ============================================
        # Onset Head
        # ============================================
        # Conv before pooling to preserve spatial features
        self.onset_head = nn.Sequential(
            nn.Conv2d(c1, c1 * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c1 * 2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),  # Global pooling
            nn.Flatten(),  # [B, c1*2]
            nn.Linear(c1 * 2, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, n_strings),
            nn.Sigmoid()
        )
        
        # ============================================
        # Pitch Head
        # ============================================
        # Direct global pooling for translation invariance
        self.pitch_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Global pooling
            nn.Flatten(),  # [B, c1]
            nn.Linear(c1, head_hidden),
            nn.LeakyReLU(0.01),  # LeakyReLU for quiet notes
            nn.Dropout(dropout),
            nn.Linear(head_hidden, c1),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout * 0.67),  # Lower dropout for final layer
            nn.Linear(c1, n_strings)
            # No activation - regression output
        )
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass through dual-branch architecture.
        
        Args:
            x: CQT spectrogram [B, 1, 13, 72]
        
        Returns:
            onset_pred: [B, 6] - onset probabilities
            pitch_pred: [B, 6] - normalized MIDI pitch
        """
        # Shared encoder
        x_shared = self.shared_encoder(x)  # [B, c1, 13, 72]
        
        # ============================================
        # Onset Branch with Residual
        # ============================================
        onset_features = self.onset_branch(x_shared)
        # Residual needs pooling to match dimensions
        onset_residual = self.onset_residual(x_shared)
        onset_residual = nn.functional.max_pool2d(onset_residual, kernel_size=(1, 2))
        onset_features = onset_features + onset_residual  # Residual connection
        onset_features = nn.functional.relu(onset_features)
        
        # ============================================
        # Pitch Branch with Residual
        # ============================================
        pitch_features = self.pitch_branch(x_shared)
        # Residual needs pooling to match dimensions
        pitch_residual = self.pitch_residual(x_shared)
        pitch_residual = nn.functional.max_pool2d(pitch_residual, kernel_size=(1, 2))
        pitch_features = pitch_features + pitch_residual  # Residual connection
        pitch_features = nn.functional.relu(pitch_features)
        
        # ============================================
        # Fusion
        # ============================================
        # Concatenate branches along channel dimension
        fused = torch.cat([onset_features, pitch_features], dim=1)  # [B, 2*c1, 13, 36]
        fused = self.fusion(fused)  # [B, c1, 13, 36]
        
        # ============================================
        # Heads
        # ============================================
        onset_pred = self.onset_head(fused)
        pitch_pred = self.pitch_head(fused)
        
        return onset_pred, pitch_pred
    
    def predict(self, x: torch.Tensor, onset_threshold: float = 0.5) -> dict:
        """
        Inference mode with post-processing.
        
        Args:
            x: CQT spectrogram [B, 1, 13, 72]
            onset_threshold: threshold for onset detection
        
        Returns:
            Dictionary with predictions
        """
        self.eval()
        with torch.no_grad():
            onset_pred, pitch_pred = self.forward(x)
            
            # Apply threshold to get binary onset predictions
            onset_binary = (onset_pred > onset_threshold).float()
            
            # Mask pitch predictions by onset
            pitch_masked = pitch_pred * onset_binary
            
            return {
                'onset_prob': onset_pred,
                'onset_binary': onset_binary,
                'pitch_midi': pitch_masked,
            }
    
    def count_parameters(self) -> int:
        """Count total number of parameters."""
        return sum(p.numel() for p in self.parameters())


def get_default_config() -> dict:
    """Get default model configuration."""
    return {
        'encoder_channels': [32, 64, 96],
        'head_hidden': 64,
        'dropout': 0.3,
        'se_reduction': 8
    }


def test_model():
    """Test the model with dummy input."""
    # Test with default parameters
    model = DualBranchCNN(**get_default_config())
    
    # Dummy input: batch=2, channels=1, time=13, freq=72
    x = torch.randn(2, 1, 13, 72)
    
    onset, pitch = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Onset output shape: {onset.shape}")
    print(f"Pitch output shape: {pitch.shape}")
    print(f"Total parameters: {model.count_parameters():,}")
    
    # Test predict mode
    result = model.predict(x)
    print(f"\nPredict mode:")
    print(f"  onset_prob shape: {result['onset_prob'].shape}")
    print(f"  onset_binary shape: {result['onset_binary'].shape}")
    print(f"  pitch_midi shape: {result['pitch_midi'].shape}")
    
    # Verify parameter count (~106K expected)
    expected_params = 106000
    actual_params = model.count_parameters()
    print(f"\nExpected parameters: ~{expected_params:,}")
    print(f"Actual parameters: {actual_params:,}")
    print(f"Difference: {abs(actual_params - expected_params) / expected_params * 100:.1f}%")


if __name__ == '__main__':
    test_model()
