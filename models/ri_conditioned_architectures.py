#!/usr/bin/env python3
"""
Richardson-Conditioned Multi-Task Networks (Matching Experimental Head Dimensions)
==================================================================================

Drop-in replacements for MLP/ResMLP/TabTransformer that add Richardson conditioning
while matching EXACT head architectures from train_*.py scripts.

Key differences from train_*.py:
- Coefficient heads: (256+1) → 128 → 64 → 1 (vs 256 → 128 → 64 → 1)
- Richardson head predicts first, then concatenated to shared features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


# ==================== SHARED COMPONENTS ====================

class RichardsonConditionedHeads(nn.Module):
    """
    Prediction heads with Richardson conditioning.
    Matches EXACT head dimensions from train_new_coeff.py: 256→128→64→output
    """
    def __init__(self, feature_dim: int):
        super().__init__()
        
        # Richardson prediction head (predicts FROM backbone features)
        # Architecture: 256 → 128 → 64 → 1 (matching train_new_coeff.py)
        self.richardson_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Regime classification head (uses backbone + Ri)
        # Architecture: 257 → 128 → 64 → 3 (extra +1 for Ri)
        self.regime_head = nn.Sequential(
            nn.Linear(feature_dim + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        
        # Viscosity head (conditioned on Ri)
        # Architecture: 257 → 128 → 64 → 1 (extra +1 for Ri)
        self.visc_head = nn.Sequential(
            nn.Linear(feature_dim + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Diffusivity head (conditioned on Ri)
        # Architecture: 257 → 128 → 64 → 1 (extra +1 for Ri)
        self.diff_head = nn.Sequential(
            nn.Linear(feature_dim + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, backbone_features: torch.Tensor):
        """
        Forward pass with Richardson conditioning.
        
        Returns: (visc, diff, richardson, regime_logits)
        """
        # Step 1: Predict Richardson number from backbone only
        ri_pred = self.richardson_head(backbone_features)
        
        # Step 2: Concatenate Ri with backbone for coefficient prediction
        conditioned_features = torch.cat([backbone_features, ri_pred], dim=1)
        
        # Step 3: Predict coefficients using Ri-conditioned features
        visc_pred = self.visc_head(conditioned_features)
        diff_pred = self.diff_head(conditioned_features)
        regime_pred = self.regime_head(conditioned_features)
        
        return visc_pred, diff_pred, ri_pred, regime_pred


# ==================== ARCHITECTURE VARIANTS ====================

class RiConditionedMLP(nn.Module):
    """
    MLP with Richardson conditioning.
    Backbone EXACTLY matches train_new_coeff.py
    Heads match dimensions but with +1 input for Ri
    """
    def __init__(self, n_features=54):
        super().__init__()
        
        # Backbone: EXACT match to train_new_coeff.py
        self.backbone = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Richardson-conditioned heads
        self.heads = RichardsonConditionedHeads(feature_dim=256)
        
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"Created Ri-conditioned MLP with {total_params:,} parameters")
    
    def forward(self, x):
        features = self.backbone(x)
        return self.heads(features)


class ResidualBlock(nn.Module):
    """Residual block for ResMLP (EXACT match to train_resmlp.py)"""
    def __init__(self, size: int, dropout: float):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(size, size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(size, size),
        )
        self.norm = nn.LayerNorm(size)

    def forward(self, x):
        return self.norm(x + self.layers(x))


class RiConditionedResMLP(nn.Module):
    """
    Residual MLP with Richardson conditioning.
    Backbone EXACTLY matches train_resmlp.py
    """
    def __init__(self, n_features=54, embed_size=256, num_blocks=4, dropout=0.3):
        super().__init__()
        
        # Backbone: EXACT match to train_resmlp.py
        self.backbone = nn.Sequential(
            nn.Linear(n_features, embed_size),
            nn.ReLU(),
            *[ResidualBlock(embed_size, dropout) for _ in range(num_blocks)]
        )
        
        # Richardson-conditioned heads
        self.heads = RichardsonConditionedHeads(feature_dim=embed_size)
        
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"Created Ri-conditioned ResMLP with {total_params:,} parameters")
    
    def forward(self, x):
        features = self.backbone(x)
        return self.heads(features)


class TransformerBlock(nn.Module):
    """Transformer block (EXACT match to train_tab_transformer.py)"""
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x


class RiConditionedTabTransformer(nn.Module):
    """
    TabTransformer with Richardson conditioning.
    Backbone EXACTLY matches train_tab_transformer.py
    """
    def __init__(self, n_features=54, embed_dim=32, num_layers=4, num_heads=8, ff_dim=128, dropout=0.2):
        super().__init__()
        
        self.n_features = n_features
        self.embed_dim = embed_dim
        
        # Feature embedding: EXACT match
        self.feature_embedder = nn.Linear(1, embed_dim)
        
        # Transformer backbone: EXACT match
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)]
        )
        
        # Richardson-conditioned heads
        final_embedding_size = n_features * embed_dim
        self.heads = RichardsonConditionedHeads(feature_dim=final_embedding_size)
        
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"Created Ri-conditioned TabTransformer with {total_params:,} parameters")
    
    def forward(self, x):
        # Embed each feature: (batch, 54) → (batch, 54, 32)
        x = x.unsqueeze(-1)
        embeddings = self.feature_embedder(x)
        
        # Transformer layers
        for block in self.transformer_blocks:
            embeddings = block(embeddings)
        
        # Flatten for heads
        features = embeddings.view(-1, self.n_features * self.embed_dim)
        
        return self.heads(features)


# ==================== TESTING ====================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*70)
    print("Testing Richardson-Conditioned Architectures (EXACT head dimensions)")
    print("="*70 + "\n")
    
    batch_size = 32
    n_features = 54
    x = torch.randn(batch_size, n_features)
    
    models = {
        'MLP': RiConditionedMLP(n_features),
        'ResMLP': RiConditionedResMLP(n_features),
        'TabTransformer': RiConditionedTabTransformer(n_features)
    }
    
    for name, model in models.items():
        print(f"\n{name}:")
        print("-" * 50)
        
        visc, diff, ri, regime = model(x)
        
        print(f"  Viscosity shape: {visc.shape}")
        print(f"  Diffusivity shape: {diff.shape}")
        print(f"  Richardson shape: {ri.shape}")
        print(f"  Regime logits shape: {regime.shape}")
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Total parameters: {total_params:,}")
        
        # Compare with non-conditioned version
        if name == 'MLP':
            from train_new_coeff import UnifiedSGSCoefficientNetwork
            baseline = UnifiedSGSCoefficientNetwork(n_features)
            baseline_params = sum(p.numel() for p in baseline.parameters())
            extra_params = total_params - baseline_params
            print(f"  Baseline parameters: {baseline_params:,}")
            print(f"  Extra parameters from Ri conditioning: {extra_params:,} (+{100*extra_params/baseline_params:.1f}%)")
    
    print("\n✅ All models tested successfully!")
    print("\nKey difference: Coefficient heads receive [features; predicted_Ri]")
    print("Head dimensions: 257→128→64→1 (vs 256→128→64→1 baseline)")
