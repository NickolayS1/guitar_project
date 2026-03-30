# -*- coding: utf-8 -*-
"""
Multimodal CNN for guitar transcription.

Combines:
- Audio branch: Enhanced Baseline CNN (72 bins, ~196K params)
- Video branch: ConvNeXt V2 Atto + TCN (~4M params)
- Fusion: Cross-Attention

Input:
    - audio: [B, 1, 13, 72] - CQT spectrogram
    - video: [B, 7, 3, 224, 224] - 7 video frames

Output:
    - onset: [B, 6] - sigmoid probabilities
    - pitch: [B, 6] - normalized MIDI
"""

import torch
import torch.nn as nn

# Handle both module and script execution
try:
    from .video_branch import VideoBranch
    from .cross_attention import CrossAttentionFusion
except ImportError:
    from video_branch import VideoBranch
    from cross_attention import CrossAttentionFusion


# ============================================================================
# Audio Branch (Enhanced Baseline CNN - 72 bins)
# ============================================================================

class SEBlock(nn.Module):
    """Channel-wise Squeeze-and-Excitation block."""
    
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
        B, C, _, _ = x.size()
        w = self.avg_pool(x).view(B, C)
        w = self.fc(w).view(B, C, 1, 1)
        return x * w


class AudioBranch(nn.Module):
    """
    Enhanced Baseline CNN for audio (72 bins).
    
    Architecture:
    - Block 1: Conv + Pool
    - Block 2: Conv + Dilated Conv + SE + Pool
    - Block 3: Conv + Conv + Residual
    """
    
    def __init__(self, dropout: float = 0.3):
        super().__init__()
        
        # Block 1: [B, 1, 13, 72] → [B, 64, 6, 36]
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        # Block 2: [B, 64, 6, 36] → [B, 64, 3, 18]
        self.block2_conv1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.block2_conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.block2_se = SEBlock(64, reduction=8)
        self.block2_pool = nn.MaxPool2d(2)
        
        # Block 3: [B, 64, 3, 18] → [B, 64, 3, 18]
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
        
        self.block3_residual = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.BatchNorm2d(64)
        )
        
        # Global pooling + projection
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout)
        
        # Project to 512 for fusion
        self.projection = nn.Sequential(
            nn.Linear(64, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1
        x = self.block1(x)
        
        # Block 2
        x = self.block2_conv1(x)
        x = self.block2_conv2(x)
        x = self.block2_se(x)
        x = self.block2_pool(x)
        
        # Block 3
        residual = self.block3_residual(x)
        x = self.block3_conv1(x)
        x = self.block3_conv2(x)
        x = x + residual
        x = nn.functional.relu(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        
        # Projection
        audio_features = self.projection(x)  # [B, 512]
        
        return audio_features


# ============================================================================
# Multimodal CNN
# ============================================================================

class MultimodalCNN(nn.Module):
    """
    Multimodal CNN with audio and video branches.
    """
    
    def __init__(
        self,
        n_strings: int = 6,
        dropout: float = 0.3,
        fusion_type: str = 'cross_attention',  # 'cross_attention' or 'simple'
        freeze_video_backbone: bool = False
    ):
        super().__init__()
        
        self.n_strings = n_strings
        self.fusion_type = fusion_type
        
        # Audio branch
        self.audio_branch = AudioBranch(dropout=dropout)
        
        # Video branch
        self.video_branch = VideoBranch(
            pretrained=True,
            freeze_backbone=freeze_video_backbone
        )
        
        # Fusion
        if fusion_type == 'cross_attention':
            self.fusion = CrossAttentionFusion(
                audio_dim=512,
                video_dim=512,
                num_heads=8,
                dropout=0.1
            )
        else:  # simple
            self.fusion = nn.Sequential(
                nn.Linear(1024, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            )
        
        # Heads
        self.onset_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, n_strings),
            nn.Sigmoid()
        )
        
        self.pitch_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, n_strings)
        )
    
    def forward(
        self,
        audio: torch.Tensor,
        video: torch.Tensor
    ) -> tuple:
        """
        Args:
            audio: [B, 1, 13, 72] - CQT spectrogram
            video: [B, 7, 3, 224, 224] - video frames
        
        Returns:
            onset: [B, 6] - onset probabilities
            pitch: [B, 6] - normalized MIDI
        """
        # Encode audio and video
        audio_features = self.audio_branch(audio)  # [B, 512]
        video_features = self.video_branch(video)  # [B, 512]
        
        # Fuse
        if self.fusion_type == 'cross_attention':
            fused = self.fusion(audio_features, video_features)  # [B, 512]
        else:
            combined = torch.cat([audio_features, video_features], dim=-1)  # [B, 1024]
            fused = self.fusion(combined)  # [B, 512]
        
        # Heads
        onset = self.onset_head(fused)
        pitch = self.pitch_head(fused)
        
        return onset, pitch
    
    def predict(
        self,
        audio: torch.Tensor,
        video: torch.Tensor,
        onset_threshold: float = 0.5
    ) -> dict:
        """Inference mode."""
        self.eval()
        with torch.no_grad():
            onset, pitch = self.forward(audio, video)
            
            onset_binary = (onset > onset_threshold).float()
            pitch_masked = pitch * onset_binary
            
            return {
                'onset_prob': onset,
                'onset_binary': onset_binary,
                'pitch_midi': pitch_masked,
            }
    
    def count_parameters(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def load_audio_weights(self, checkpoint_path: str):
        """
        Load pretrained audio branch weights.
        
        Args:
            checkpoint_path: Path to enhanced_baseline checkpoint
        """
        import torch
        
        print(f"Loading audio weights from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load full state dict
        full_state = checkpoint['model_state_dict']
        
        # Audio branch keys match exactly (no prefix needed)
        # Just load directly
        missing_keys, unexpected_keys = self.audio_branch.load_state_dict(full_state, strict=False)
        
        if missing_keys:
            print(f"  Missing keys: {len(missing_keys)}")
        if unexpected_keys:
            print(f"  Unexpected keys: {len(unexpected_keys)}")
        
        print(f"  Loaded {len(full_state)} parameters")
        
        # Freeze audio branch
        for param in self.audio_branch.parameters():
            param.requires_grad = False
        
        print(f"  Audio branch frozen")
    
    def freeze_video_backbone(self):
        """Freeze video branch backbone."""
        self.video_branch.freeze_backbone()
    
    def unfreeze_video_backbone(self):
        """Unfreeze video branch backbone."""
        self.video_branch.unfreeze_backbone()
    
    def freeze_audio_branch(self):
        """Freeze audio branch."""
        for param in self.audio_branch.parameters():
            param.requires_grad = False
    
    def unfreeze_audio_branch(self):
        """Unfreeze audio branch."""
        for param in self.audio_branch.parameters():
            param.requires_grad = True


def test_model():
    """Test the model with dummy input."""
    model = MultimodalCNN(
        fusion_type='cross_attention',
        freeze_video_backbone=True  # Start frozen
    )
    
    # Dummy inputs
    audio = torch.randn(2, 1, 13, 72)
    video = torch.randn(2, 7, 3, 224, 224)
    
    onset, pitch = model(audio, video)
    
    print(f"Audio input: {audio.shape}")
    print(f"Video input: {video.shape}")
    print(f"Onset output: {onset.shape}")
    print(f"Pitch output: {pitch.shape}")
    print(f"\nTotal parameters: {model.count_parameters():,}")
    
    # Test predict mode
    result = model.predict(audio, video)
    print(f"\nPredict mode:")
    print(f"  onset_prob: {result['onset_prob'].shape}")
    print(f"  onset_binary: {result['onset_binary'].shape}")
    print(f"  pitch_midi: {result['pitch_midi'].shape}")
    
    # Test freeze/unfreeze
    print(f"\nVideo backbone frozen: {not any(p.requires_grad for p in model.video_branch.backbone.parameters())}")
    model.unfreeze_video_backbone()
    print(f"Video backbone unfrozen: {any(p.requires_grad for p in model.video_branch.backbone.parameters())}")


if __name__ == '__main__':
    test_model()
