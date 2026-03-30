# -*- coding: utf-8 -*-
"""
Baseline CNN model for guitar transcription.

Late-split architecture with shared encoder and dual prediction heads.
~250K parameters (optimized baseline).

Input: [B, 1, 13, 72] - CQT spectrogram
Output: 
    - onset_pred: [B, 6] - sigmoid probabilities
    - pitch_pred: [B, 6] - normalized MIDI (0-1 range)
"""

import torch
import torch.nn as nn


class BaselineCNN(nn.Module):
    """
    Baseline CNN with late split for guitar transcription.
    
    Architecture (optimized for ~250K parameters):
    - Shared encoder: 3 conv blocks with reduced channels [32, 64, 128]
    - Dual heads: Onset (classification) + Pitch (regression) with reduced hidden size
    """
    
    def __init__(self, n_strings: int = 6, encoder_channels: list = None, head_hidden: int = 64, dropout: float = 0.3):
        super().__init__()
        
        self.n_strings = n_strings
        
        # Default channels if not specified
        if encoder_channels is None:
            encoder_channels = [32, 64, 128]
        
        c1, c2, c3 = encoder_channels
        
        # === Shared Encoder ===
        # Block 1: [B, 1, 13, 72] → [B, c1, 6, 36]
        self.encoder1 = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c1, kernel_size=3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # [B, c1, 6, 36]
        )
        
        # Block 2: [B, c1, 6, 36] → [B, c2, 3, 18]
        self.encoder2 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # [B, c2, 3, 18]
        )
        
        # Block 3: [B, c2, 3, 18] → [B, c3, 3, 18]
        self.encoder3 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3, c3, kernel_size=3, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True)
            # No pooling - keep spatial dimensions
        )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # [B, c3, 1, 1]
        
        # Flatten
        self.flatten = nn.Flatten()  # [B, c3]
        
        # Dropout before heads
        self.dropout = nn.Dropout(dropout)
        
        # === Onset Head ===
        self.onset_head = nn.Sequential(
            nn.Linear(c3, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, n_strings),
            nn.Sigmoid()
        )
        
        # === Pitch Head ===
        self.pitch_head = nn.Sequential(
            nn.Linear(c3, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, n_strings)
            # No activation - regression output
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
        # Shared encoder
        x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.encoder3(x)
        
        # Global pooling
        x = self.global_pool(x)
        
        # Flatten
        x = self.flatten(x)
        
        # Dropout
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


def get_default_config() -> dict:
    """Get default model configuration."""
    return {
        'encoder_channels': [32, 64, 96],
        'head_hidden': 48,
        'dropout': 0.3
    }


def test_model():
    """Test the model with dummy input."""
    # Test with default parameters (optimized baseline)
    model = BaselineCNN(
        encoder_channels=[32, 64, 96],
        head_hidden=48
    )
    
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
