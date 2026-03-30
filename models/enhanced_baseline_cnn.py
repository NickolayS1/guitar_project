# -*- coding: utf-8 -*-
"""
Enhanced Baseline CNN model for guitar transcription.

Architecture improvements over baseline:
1. SE Block after encoder block 2 (feature selection)
2. Dilated convolution (dilation=2) in block 2 (preserve information)
3. Residual connection in block 3 (improve gradient flow)

~240K parameters (vs 214K baseline).

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


class EnhancedBaselineCNN(nn.Module):
    """
    Enhanced Baseline CNN with SE block, dilated conv, and residual connection.
    
    Architecture:
    - Block 1: Standard conv + pool
    - Block 2: Conv + Dilated Conv (dilation=2) + SE Block + pool
    - Block 3: Conv + Conv + Residual Connection
    - Dual heads: Onset (classification) + Pitch (regression)
    """
    
    def __init__(self, n_strings: int = 6, dropout: float = 0.3):
        super().__init__()
        
        self.n_strings = n_strings
        
        # ============================================
        # Block 1: Standard conv + pool
        # [B, 1, 13, 72] → [B, 64, 6, 36]
        # ============================================
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # [B, 64, 6, 36]
        )
        
        # ============================================
        # Block 2: Conv + Dilated Conv + SE + pool
        # [B, 64, 6, 36] → [B, 64, 3, 18]
        # ============================================
        self.block2_conv1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Dilated convolution (dilation=2) - preserves spatial info
        self.block2_conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # SE Block for feature selection
        self.block2_se = SEBlock(64, reduction=8)
        
        self.block2_pool = nn.MaxPool2d(2)  # [B, 64, 3, 18]
        
        # ============================================
        # Block 3: Conv + Conv + Residual
        # [B, 64, 3, 18] → [B, 64, 3, 18]
        # ============================================
        self.block3_conv1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.block3_conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Residual connection (no channel change)
        self.block3_residual = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.BatchNorm2d(64)
        )
        
        # ============================================
        # Global pooling + Heads
        # ============================================
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # [B, 128, 1, 1]
        self.flatten = nn.Flatten()  # [B, 128]
        self.dropout = nn.Dropout(dropout)
        
        # Onset Head
        self.onset_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, n_strings),
            nn.Sigmoid()
        )
        
        # Pitch Head
        self.pitch_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, n_strings)
        )
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass.
        
        Args:
            x: CQT spectrogram [B, 1, 13, 72]
        
        Returns:
            onset_pred: [B, 6] - onset probabilities
            pitch_pred: [B, 6] - normalized MIDI pitch
        """
        # Block 1
        x = self.block1(x)  # [B, 64, 6, 36]
        
        # Block 2
        x = self.block2_conv1(x)  # [B, 64, 6, 36]
        x = self.block2_conv2(x)  # [B, 64, 6, 36] (dilated)
        x = self.block2_se(x)  # SE attention
        x = self.block2_pool(x)  # [B, 64, 3, 18]
        
        # Block 3 with residual
        residual = self.block3_residual(x)  # [B, 64, 3, 18]
        x = self.block3_conv1(x)  # [B, 64, 3, 18]
        x = self.block3_conv2(x)  # [B, 64, 3, 18]
        x = x + residual  # Residual connection
        x = nn.functional.relu(x)
        
        # Global pooling
        x = self.global_pool(x)  # [B, 64, 1, 1]
        x = self.flatten(x)  # [B, 64]
        x = self.dropout(x)
        
        # Heads
        onset_pred = self.onset_head(x)
        pitch_pred = self.pitch_head(x)
        
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


def test_model():
    """Test the model with dummy input."""
    model = EnhancedBaselineCNN()
    
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


if __name__ == '__main__':
    test_model()
