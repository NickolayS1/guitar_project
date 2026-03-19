# -*- coding: utf-8 -*-
"""
Baseline CNN model for guitar transcription.

Late-split architecture with shared encoder and dual prediction heads.
~266K parameters.

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
    
    Architecture:
    - Shared encoder: 3 conv blocks with max pooling
    - Dual heads: Onset (classification) + Pitch (regression)
    """
    
    def __init__(self, n_strings: int = 6, dropout: float = 0.3):
        super().__init__()
        
        self.n_strings = n_strings
        
        # === Shared Encoder ===
        # Block 1: [B, 1, 13, 72] → [B, 64, 6, 36]
        self.encoder1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # [B, 64, 6, 36]
        )
        
        # Block 2: [B, 64, 6, 36] → [B, 128, 3, 18]
        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # [B, 128, 3, 18]
        )
        
        # Block 3: [B, 128, 3, 18] → [B, 256, 3, 18]
        self.encoder3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
            # No pooling - keep spatial dimensions
        )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # [B, 256, 1, 1]
        
        # Flatten
        self.flatten = nn.Flatten()  # [B, 256]
        
        # Dropout before heads
        self.dropout = nn.Dropout(dropout)
        
        # === Onset Head ===
        self.onset_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_strings),
            nn.Sigmoid()
        )
        
        # === Pitch Head ===
        self.pitch_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_strings)
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


def test_model():
    """Test the model with dummy input."""
    model = BaselineCNN()
    
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
