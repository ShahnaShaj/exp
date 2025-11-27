"""
Core Model Architecture for EEG2TEXT
Implements: CNN Compressor, Transformer Encoder, Multi-View Transformer, and complete EEG2TEXT model
Based on: "Open Vocabulary EEG-to-Text Decoding with EEG Pre-Training"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BartForConditionalGeneration, BartConfig
from config import Config


# ============================================================================
# 1. CNN MODULE FOR EEG COMPRESSION
# ============================================================================

class CNNCompressor(nn.Module):
    """
    CNN module to compress long EEG signals temporally
    Input: (batch, time_steps, channels)
    Output: (batch, compressed_time, hidden_dim)
    """
    
    def __init__(self, num_channels=105, hidden_dim=512):
        super().__init__()
        
        self.num_channels = num_channels
        self.hidden_dim = hidden_dim
        
        # Temporal compression with stride=2 to reduce sequence length by 4x
        self.temporal_conv = nn.Sequential(
            # First conv layer: reduce by 2x
            nn.Conv1d(
                in_channels=num_channels,
                out_channels=hidden_dim // 2,
                kernel_size=5,
                stride=2,
                padding=2
            ),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # Second conv layer: reduce by 2x (total 4x compression)
            nn.Conv1d(
                in_channels=hidden_dim // 2,
                out_channels=hidden_dim,
                kernel_size=5,
                stride=2,
                padding=2
            ),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, time_steps, channels)
        Returns:
            (batch, compressed_time, hidden_dim)
        """
        # Transpose for Conv1d: (batch, channels, time_steps)
        x = x.transpose(1, 2)
        
        # Apply temporal convolution
        x = self.temporal_conv(x)
        
        # Transpose back: (batch, compressed_time, hidden_dim)
        x = x.transpose(1, 2)
        
        return x


# ============================================================================
# 2. TRANSFORMER ENCODER
# ============================================================================

class TransformerEncoder(nn.Module):
    """
    Transformer encoder for EEG feature extraction
    """
    
    def __init__(
        self,
        hidden_dim=512,
        num_layers=6,
        num_heads=8,
        ff_dim=2048,
        dropout=0.1
    ):
        super().__init__()
        
        # Create transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=False  # Post-norm
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, hidden_dim)
            mask: (batch, seq_len) - True for positions to mask
        Returns:
            (batch, seq_len, hidden_dim)
        """
        x = self.transformer(x, src_key_padding_mask=mask)
        x = self.layer_norm(x)
        return x


# ============================================================================
# 3. CONVOLUTIONAL TRANSFORMER (BASE MODEL)
# ============================================================================

class ConvolutionalTransformer(nn.Module):
    """
    Complete base model: CNN + Transformer
    This serves as the foundation for both pre-training and the multi-view model
    """
    
    def __init__(
        self,
        num_channels=105,
        hidden_dim=512,
        num_layers=6,
        num_heads=8,
        ff_dim=2048,
        dropout=0.1
    ):
        super().__init__()
        
        self.cnn = CNNCompressor(num_channels, hidden_dim)
        self.transformer = TransformerEncoder(
            hidden_dim, num_layers, num_heads, ff_dim, dropout
        )
        
        self.hidden_dim = hidden_dim
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, time_steps, channels)
            mask: Optional padding mask
        Returns:
            (batch, compressed_time, hidden_dim)
        """
        # CNN compression
        x = self.cnn(x)
        
        # Transformer encoding
        x = self.transformer(x, mask)
        
        return x


# ============================================================================
# 4. PRE-TRAINING MODULE
# ============================================================================

class EEGPreTraining(nn.Module):
    """
    Self-supervised pre-training module with masked EEG reconstruction
    Implements the pre-training strategy from the paper
    """
    
    def __init__(
        self,
        num_channels=105,
        hidden_dim=512,
        num_layers=6,
        num_heads=8,
        ff_dim=2048,
        dropout=0.1
    ):
        super().__init__()
        
        self.num_channels = num_channels
        
        # Encoder
        self.encoder = ConvolutionalTransformer(
            num_channels, hidden_dim, num_layers, num_heads, ff_dim, dropout
        )
        
        # Decoder for reconstruction
        self.decoder = nn.Sequential(
            # Upsample back to original time dimension
            nn.ConvTranspose1d(
                hidden_dim,
                hidden_dim,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1
            ),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.ConvTranspose1d(
                hidden_dim,
                num_channels,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1
            )
        )
    
    def mask_eeg(self, x, mask_ratio=0.15, mask_strategy='remask'):
        """
        Apply masking strategies to EEG input
        
        Args:
            x: (batch, time_steps, channels)
            mask_ratio: Fraction of data to mask
            mask_strategy: 'remask', 'masked', or 'continuous'
        
        Returns:
            masked_x: EEG with masked regions
            mask: Binary mask (1 = keep, 0 = masked)
        """
        batch_size, time_steps, channels = x.shape
        device = x.device
        
        if mask_strategy == 'remask':
            # Re-randomize masking each time (best performance in paper)
            mask = torch.rand(batch_size, time_steps, 1, device=device) > mask_ratio
            
        elif mask_strategy == 'continuous':
            # Mask continuous spans
            mask = torch.ones(batch_size, time_steps, 1, device=device)
            span_length = int(time_steps * mask_ratio)
            
            for b in range(batch_size):
                if span_length > 0:
                    start = torch.randint(0, time_steps - span_length + 1, (1,)).item()
                    mask[b, start:start+span_length, :] = 0
        
        else:  # 'masked' - fixed random mask
            mask = torch.rand(batch_size, time_steps, 1, device=device) > mask_ratio
        
        # Apply mask (set masked positions to zero)
        masked_x = x * mask
        
        return masked_x, mask
    
    def forward(self, x, mask_ratio=0.15, mask_strategy='remask', return_details=False):
        """
        Forward pass with masking and reconstruction
        
        Args:
            x: (batch, time_steps, channels)
            mask_ratio: Fraction to mask
            mask_strategy: Masking strategy
            return_details: If True, return all intermediate outputs
        
        Returns:
            If return_details=False:
                loss: Reconstruction loss
            If return_details=True:
                loss, reconstructed, masked_x, mask
        """
        # Apply masking
        masked_x, mask = self.mask_eeg(x, mask_ratio, mask_strategy)
        
        # Encode
        encoded = self.encoder(masked_x)
        
        # Decode for reconstruction
        # Need to transpose for ConvTranspose1d
        encoded = encoded.transpose(1, 2)  # (batch, hidden_dim, compressed_time)
        reconstructed = self.decoder(encoded)  # (batch, channels, time_steps)
        reconstructed = reconstructed.transpose(1, 2)  # (batch, time_steps, channels)
        
        # Handle potential size mismatch due to convolution operations
        if reconstructed.shape[1] != x.shape[1]:
            # Adjust to match original size
            if reconstructed.shape[1] > x.shape[1]:
                reconstructed = reconstructed[:, :x.shape[1], :]
            else:
                # Pad if needed
                pad_len = x.shape[1] - reconstructed.shape[1]
                reconstructed = F.pad(reconstructed, (0, 0, 0, pad_len))
        
        # Compute reconstruction loss only on masked positions
        # Invert mask: we want loss on the masked parts (where mask=True)
        loss_mask = ~mask  # Use logical NOT for boolean tensors
        loss = F.mse_loss(reconstructed * loss_mask, x * loss_mask, reduction='sum')
        loss = loss / (loss_mask.sum() + 1e-8)  # Normalize by number of masked elements
        
        if return_details:
            return loss, reconstructed, masked_x, mask
        return loss


# ============================================================================
# 5. MULTI-VIEW TRANSFORMER
# ============================================================================

class MultiViewTransformer(nn.Module):
    """
    Multi-view transformer with separate encoders for different brain regions
    Implements the region-specific encoding from Table 1 in the paper
    """
    
    def __init__(
        self,
        channel_groups,
        hidden_dim=512,
        num_layers=6,
        num_heads=8,
        ff_dim=2048,
        dropout=0.1
    ):
        super().__init__()
        
        self.channel_groups = channel_groups
        self.region_names = list(channel_groups.keys())
        self.num_regions = len(self.region_names)
        
        # Create separate encoder for each brain region
        self.region_encoders = nn.ModuleDict({
            region: ConvolutionalTransformer(
                num_channels=len(channels),
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout
            )
            for region, channels in channel_groups.items()
        })
        
        # Global transformer to combine all regions
        self.global_transformer = TransformerEncoder(
            hidden_dim=hidden_dim * self.num_regions,
            num_layers=3,  # Fewer layers for global combination
            num_heads=num_heads,
            ff_dim=ff_dim * self.num_regions,
            dropout=dropout
        )
        
        # Projection layer to target dimension
        self.projection = nn.Linear(hidden_dim * self.num_regions, hidden_dim)
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x):
        """
        Args:
            x: (batch, time_steps, all_channels)
        Returns:
            (batch, compressed_time, hidden_dim)
        """
        # Extract features from each brain region
        region_features = []
        
        for region, channels in self.channel_groups.items():
            # Select channels for this region
            region_eeg = x[:, :, channels]
            
            # Encode this region
            region_feat = self.region_encoders[region](region_eeg)
            region_features.append(region_feat)
        
        # Ensure all features have the same sequence length
        # (they should after CNN compression, but just to be safe)
        min_len = min(feat.shape[1] for feat in region_features)
        region_features = [feat[:, :min_len, :] for feat in region_features]
        
        # Concatenate all region features along feature dimension
        combined = torch.cat(region_features, dim=-1)
        
        # Global transformer to model inter-region dependencies
        global_feat = self.global_transformer(combined)
        
        # Project to target dimension
        output = self.projection(global_feat)
        output = self.layer_norm(output)
        
        return output


# ============================================================================
# 6. COMPLETE EEG2TEXT MODEL
# ============================================================================

# ============================================================================
# 6. CROSS-MODAL BRIDGE
# ============================================================================

class CrossModalBridge(nn.Module):
    """
    Cross-modal attention bridge between EEG and BART
    Allows BART decoder to attend to relevant EEG features
    """
    
    def __init__(self, eeg_dim, bart_dim, num_heads=8, num_layers=2, dropout=0.1):
        super().__init__()
        
        self.eeg_dim = eeg_dim
        self.bart_dim = bart_dim
        
        # Project EEG to BART dimension for keys/values
        self.eeg_proj = nn.Linear(eeg_dim, bart_dim)
        
        # Stack of cross-attention layers
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=bart_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        
        # Layer norms
        self.norms = nn.ModuleList([
            nn.LayerNorm(bart_dim)
            for _ in range(num_layers)
        ])
        
        # Feed-forward networks
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(bart_dim, bart_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(bart_dim * 4, bart_dim),
                nn.Dropout(dropout)
            )
            for _ in range(num_layers)
        ])
        
        self.ffn_norms = nn.ModuleList([
            nn.LayerNorm(bart_dim)
            for _ in range(num_layers)
        ])
    
    def forward(self, eeg_features, bart_queries=None):
        """
        Cross-modal attention from BART to EEG
        
        Args:
            eeg_features: (batch, eeg_seq, eeg_dim)
            bart_queries: (batch, text_seq, bart_dim) - if None, uses projected EEG
        
        Returns:
            attended: (batch, text_seq, bart_dim)
            attn_weights: List of attention weight matrices
        """
        # Project EEG features
        eeg_proj = self.eeg_proj(eeg_features)  # [batch, eeg_seq, bart_dim]
        
        # If no queries provided, use projected EEG as queries (self-attention style)
        if bart_queries is None:
            bart_queries = eeg_proj
        
        attn_weights_list = []
        x = bart_queries
        
        # Apply cross-attention layers
        for cross_attn, norm, ffn, ffn_norm in zip(
            self.cross_attn_layers, self.norms, self.ffns, self.ffn_norms
        ):
            # Cross-attention
            attended, attn_weights = cross_attn(
                query=x,
                key=eeg_proj,
                value=eeg_proj
            )
            
            # Residual + norm
            x = norm(x + attended)
            
            # FFN
            x = ffn_norm(x + ffn(x))
            
            attn_weights_list.append(attn_weights)
        
        return x, attn_weights_list


class SimpleProjection(nn.Module):
    """
    Simple linear projection from EEG to BART dimension
    Memory-efficient baseline
    """
    
    def __init__(self, eeg_dim, bart_dim):
        super().__init__()
        self.proj = nn.Linear(eeg_dim, bart_dim)
    
    def forward(self, eeg_features, bart_queries=None):
        """
        Args:
            eeg_features: (batch, seq, eeg_dim)
            bart_queries: Ignored (for API compatibility)
        
        Returns:
            projected: (batch, seq, bart_dim)
            attn_weights: None (no attention)
        """
        return self.proj(eeg_features), None


# ============================================================================
# 7. COMPLETE EEG2TEXT MODEL
# ============================================================================

class EEG2TEXT(nn.Module):
    """
    Complete EEG2TEXT model: Multi-view Transformer + BART
    """
    
    def __init__(
        self,
        channel_groups,
        hidden_dim=512,
        num_layers=6,
        num_heads=8,
        ff_dim=2048,
        dropout=0.1,
        bart_model='facebook/bart-base'
    ):
        super().__init__()
        
        # Multi-view EEG encoder
        self.eeg_encoder = MultiViewTransformer(
            channel_groups=channel_groups,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout
        )
        
        print("  Loading BART model (this may take a while on CPU)...")
        # Load pre-trained BART with low memory usage
        self.bart = BartForConditionalGeneration.from_pretrained(
            bart_model,
            low_cpu_mem_usage=True  # Reduce memory footprint
        )
        print("  ✓ BART loaded")
        
        # Apply LoRA if enabled (reduces trainable params by 98%)
        from config import Config
        if hasattr(Config, 'USE_LORA') and Config.USE_LORA:
            from peft import LoraConfig, get_peft_model
            
            print("  Applying LoRA to BART...")
            lora_config = LoraConfig(
                r=Config.LORA_R,
                lora_alpha=Config.LORA_ALPHA,
                target_modules=Config.LORA_TARGET_MODULES,
                lora_dropout=Config.LORA_DROPOUT,
                bias="none",
                task_type="SEQ_2_SEQ_LM"
            )
            
            self.bart = get_peft_model(self.bart, lora_config)
            self.bart.print_trainable_parameters()
            print("  ✓ LoRA applied - 98% memory reduction!")
        else:
            trainable = sum(p.numel() for p in self.bart.parameters() if p.requires_grad)
            print(f"  BART trainable params: {trainable:,} (full fine-tuning)")
        
        # Projection to match BART's input dimension
        bart_dim = self.bart.config.d_model
        self.proj_to_bart = nn.Linear(hidden_dim, bart_dim)
        
        self.hidden_dim = hidden_dim
        self.bart_dim = bart_dim
    
    def forward(self, eeg, decoder_input_ids=None, decoder_attention_mask=None, labels=None):
        """
        Forward pass for training
        
        Args:
            eeg: (batch, time_steps, channels)
            decoder_input_ids: (batch, seq_len) - for teacher forcing
            decoder_attention_mask: (batch, seq_len)
            labels: (batch, seq_len) - target tokens
        
        Returns:
            BartOutput with loss and logits
        """
        # Encode EEG
        eeg_features = self.eeg_encoder(eeg)
        
        # Project to BART dimension
        eeg_features = self.proj_to_bart(eeg_features)
        
        # Create attention mask for EEG features (all valid)
        encoder_attention_mask = torch.ones(
            eeg_features.shape[0],
            eeg_features.shape[1],
            dtype=torch.long,
            device=eeg_features.device
        )
        
        # Use BART with EEG features as encoder outputs
        outputs = self.bart(
            inputs_embeds=eeg_features,
            attention_mask=encoder_attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            return_dict=True
        )
        
        return outputs
    
    def generate(self, eeg, max_length=50, num_beams=5, **kwargs):
        """
        Generate text from EEG signals
        
        Args:
            eeg: (batch, time_steps, channels)
            max_length: Maximum generation length
            num_beams: Number of beams for beam search
        
        Returns:
            Generated token IDs (batch, seq_len)
        """
        # Encode EEG
        eeg_features = self.eeg_encoder(eeg)
        
        # Project to BART dimension
        eeg_features = self.proj_to_bart(eeg_features)
        
        # Create attention mask
        encoder_attention_mask = torch.ones(
            eeg_features.shape[0],
            eeg_features.shape[1],
            dtype=torch.long,
            device=eeg_features.device
        )
        
        # Generate with BART
        outputs = self.bart.generate(
            inputs_embeds=eeg_features,
            attention_mask=encoder_attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
            **kwargs
        )
        
        return outputs
    
    def load_pretrained_encoder(self, pretrained_path):
        """
        Load pre-trained encoder weights into multi-view encoders
        
        Args:
            pretrained_path: Path to pre-trained model checkpoint
        """
        pretrained_state = torch.load(pretrained_path, map_location='cpu')
        
        # If the saved model is EEGPreTraining, extract encoder weights
        if 'encoder.cnn.temporal_conv.0.weight' in pretrained_state:
            encoder_state = {
                k.replace('encoder.', ''): v 
                for k, v in pretrained_state.items() 
                if k.startswith('encoder.')
            }
        else:
            encoder_state = pretrained_state
        
        # Load into each region encoder
        for region_name, region_encoder in self.eeg_encoder.region_encoders.items():
            try:
                # Load with strict=False to allow for channel dimension mismatch
                region_encoder.load_state_dict(encoder_state, strict=False)
                print(f"  ✓ Loaded pre-trained weights for region: {region_name}")
            except Exception as e:
                print(f"  ⚠ Warning: Could not load weights for {region_name}: {e}")


def test_models():
    """Test model instantiation and forward pass"""
    print("Testing model components...")
    
    batch_size = 2
    time_steps = 2000
    num_channels = 105
    
    # Test data
    x = torch.randn(batch_size, time_steps, num_channels)
    
    # Test CNN Compressor
    print("\n1. Testing CNN Compressor...")
    cnn = CNNCompressor(num_channels, hidden_dim=512)
    out = cnn(x)
    print(f"  Input: {x.shape} -> Output: {out.shape}")
    
    # Test Convolutional Transformer
    print("\n2. Testing Convolutional Transformer...")
    conv_trans = ConvolutionalTransformer(num_channels, hidden_dim=512)
    out = conv_trans(x)
    print(f"  Input: {x.shape} -> Output: {out.shape}")
    
    # Test Pre-training Module
    print("\n3. Testing Pre-training Module...")
    pretrain = EEGPreTraining(num_channels, hidden_dim=512)
    loss, reconstructed = pretrain(x)
    print(f"  Reconstruction Loss: {loss.item():.4f}")
    print(f"  Reconstructed shape: {reconstructed.shape}")
    
    # Test Multi-view Transformer
    print("\n4. Testing Multi-View Transformer...")
    channel_groups = Config.CHANNEL_GROUPS
    mv_trans = MultiViewTransformer(channel_groups, hidden_dim=512)
    out = mv_trans(x)
    print(f"  Input: {x.shape} -> Output: {out.shape}")
    
    # Test Complete EEG2TEXT
    print("\n5. Testing Complete EEG2TEXT Model...")
    model = EEG2TEXT(channel_groups, hidden_dim=512)
    
    # Dummy decoder inputs
    decoder_input_ids = torch.randint(0, 1000, (batch_size, 20))
    decoder_attention_mask = torch.ones(batch_size, 20)
    labels = torch.randint(0, 1000, (batch_size, 20))
    
    outputs = model(x, decoder_input_ids, decoder_attention_mask, labels)
    print(f"  Loss: {outputs.loss.item():.4f}")
    print(f"  Logits shape: {outputs.logits.shape}")
    
    print("\n✓ All model tests passed!")


if __name__ == '__main__':
    test_models()
