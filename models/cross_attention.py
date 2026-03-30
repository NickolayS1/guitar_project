# -*- coding: utf-8 -*-
"""
Cross-attention fusion for multimodal guitar transcription.

Audio is primary (query), Video is context (key/value).

Input:
    - audio_features: [B, 512]
    - video_features: [B, 512]

Output:
    - fused_features: [B, 512]
"""

import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention fusion where audio queries video.
    
    Architecture:
    1. Project video to audio space
    2. Multi-head cross-attention (audio queries video)
    3. Residual connection + layer norm
    4. Concat + MLP projection
    """
    
    def __init__(
        self,
        audio_dim: int = 512,
        video_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Project video to audio space
        self.video_proj = nn.Linear(video_dim, audio_dim)
        
        # Multi-head cross-attention
        # audio = query, video = key/value
        self.attention = nn.MultiheadAttention(
            embed_dim=audio_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer norm + residual
        self.norm = nn.LayerNorm(audio_dim)
        
        # Output projection (concat audio + attended_video)
        self.output_proj = nn.Sequential(
            nn.Linear(audio_dim * 2, audio_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
    
    def forward(
        self,
        audio_features: torch.Tensor,
        video_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            audio_features: [B, 512] - audio features (query)
            video_features: [B, 512] - video features (context)
        
        Returns:
            [B, 512] - fused features
        """
        # Add sequence dimension for attention
        audio_exp = audio_features.unsqueeze(1)  # [B, 1, 512]
        video_proj = self.video_proj(video_features).unsqueeze(1)  # [B, 1, 512]
        
        # Cross-attention: audio queries video
        attn_out, _ = self.attention(
            query=audio_exp,
            key=video_proj,
            value=video_proj
        )  # [B, 1, 512]
        
        # Residual connection + layer norm
        fused = self.norm(audio_features + attn_out.squeeze(1))  # [B, 512]
        
        # Concatenate audio + attended video context
        combined = torch.cat([fused, video_features], dim=-1)  # [B, 1024]
        
        # Project back to audio dimension
        output = self.output_proj(combined)  # [B, 512]
        
        return output


class SimpleFusion(nn.Module):
    """
    Simple concatenation fusion (baseline).
    
    For comparison with cross-attention.
    """
    
    def __init__(self, audio_dim: int = 512, video_dim: int = 512):
        super().__init__()
        
        self.fusion = nn.Sequential(
            nn.Linear(audio_dim + video_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
    
    def forward(
        self,
        audio_features: torch.Tensor,
        video_features: torch.Tensor
    ) -> torch.Tensor:
        """Concatenate and project."""
        combined = torch.cat([audio_features, video_features], dim=-1)
        return self.fusion(combined)


def test_fusion():
    """Test fusion modules."""
    audio = torch.randn(4, 512)
    video = torch.randn(4, 512)
    
    # Cross-attention fusion
    cross_attn = CrossAttentionFusion()
    out_ca = cross_attn(audio, video)
    print(f"Cross-Attention: {audio.shape} + {video.shape} → {out_ca.shape}")
    
    # Simple fusion
    simple = SimpleFusion()
    out_simple = simple(audio, video)
    print(f"Simple: {audio.shape} + {video.shape} → {out_simple.shape}")
    
    # Parameter count
    print(f"\nCross-Attention params: {sum(p.numel() for p in cross_attn.parameters()):,}")
    print(f"Simple params: {sum(p.numel() for p in simple.parameters()):,}")


if __name__ == '__main__':
    test_fusion()
