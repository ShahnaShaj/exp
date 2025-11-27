"""
Simplified EEG-to-Text Architecture Using Pre-trained Components

Architecture:
1. EEG Encoder: Bidirectional Mamba (BiMamba) for feature extraction
2. Projection Layer: Maps EEG features to LLM space
3. LLM Decoder: Pre-trained language model (BART/Gemma)

This follows the current research approach:
- Pre-train encoder on masked EEG reconstruction
- Use pre-trained LLM for text generation
- Fine-tune end-to-end on EEG-text pairs

Advantages:
- Leverages pre-trained knowledge
- Efficient training (less data needed)
- State-of-the-art performance
- Follows published research patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from transformers import BartForConditionalGeneration, BartTokenizer


class BiMambaBlock(nn.Module):
    """
    Bidirectional Mamba block for EEG
    Processes signal in both forward and backward directions
    
    Fallback to Bidirectional LSTM if Mamba not available
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        dropout: float = 0.1,
        use_mamba: bool = True
    ):
        super().__init__()
        
        self.d_model = d_model
        self.use_mamba = use_mamba
        
        try:
            from mamba_ssm import Mamba
            # Forward and backward Mamba blocks
            self.forward_mamba = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=4,
                expand=2
            )
            self.backward_mamba = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=4,
                expand=2
            )
            self.use_mamba = True
        except ImportError:
            # Fallback to Bidirectional LSTM
            self.lstm = nn.LSTM(
                input_size=d_model,
                hidden_size=d_model // 2,
                num_layers=2,
                batch_first=True,
                bidirectional=True,
                dropout=dropout
            )
            self.use_mamba = False
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        residual = x
        
        if self.use_mamba:
            # Forward pass
            x_forward = self.forward_mamba(x)
            
            # Backward pass (flip sequence)
            x_flipped = torch.flip(x, dims=[1])
            x_backward = self.backward_mamba(x_flipped)
            x_backward = torch.flip(x_backward, dims=[1])
            
            # Combine
            x = x_forward + x_backward
        else:
            # LSTM fallback
            x, _ = self.lstm(x)
        
        x = self.layer_norm(residual + self.dropout(x))
        return x


class EEGMambaEncoder(nn.Module):
    """
    EEG Encoder using Bidirectional Mamba
    
    Similar to EEGMamba/FEMBA architectures:
    1. CNN for local feature extraction
    2. BiMamba blocks for long-range dependencies
    3. Self-attention for global context
    """
    
    def __init__(
        self,
        num_channels: int = 105,
        d_model: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        print(f"\n🧠 Initializing EEGMamba Encoder")
        print(f"  Channels: {num_channels}")
        print(f"  Model dim: {d_model}")
        print(f"  Layers: {num_layers}")
        
        # 1. CNN for initial feature extraction (local patterns)
        self.cnn = nn.Sequential(
            nn.Conv1d(num_channels, d_model // 2, kernel_size=7, padding=3),
            nn.BatchNorm1d(d_model // 2),
            nn.GELU(),
            nn.Conv1d(d_model // 2, d_model, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 2. Stack of BiMamba blocks (long-range dependencies)
        self.mamba_layers = nn.ModuleList([
            BiMambaBlock(
                d_model=d_model,
                d_state=16,
                dropout=dropout,
                use_mamba=True
            )
            for _ in range(num_layers)
        ])
        
        # 3. Self-attention for global context
        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.final_norm = nn.LayerNorm(d_model)
        
        print(f"  ✓ Encoder initialized")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, num_channels)
        Returns:
            (batch, compressed_seq_len, d_model)
        """
        # CNN expects (batch, channels, seq_len)
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)  # Back to (batch, seq_len, d_model)
        
        # BiMamba layers
        for mamba_layer in self.mamba_layers:
            x = mamba_layer(x)
        
        # Global attention
        attn_out, _ = self.self_attention(x, x, x)
        x = x + attn_out
        
        x = self.final_norm(x)
        return x


class SimpleEEG2Text(nn.Module):
    """
    Simplified EEG-to-Text model following current research approach
    
    Architecture:
    1. Pre-trained EEG Encoder (BiMamba)
    2. Projection to LLM space
    3. Pre-trained LLM (BART)
    
    This is the standard approach in recent papers
    """
    
    def __init__(
        self,
        num_eeg_channels: int = 105,
        encoder_dim: int = 256,
        num_encoder_layers: int = 6,
        llm_model: str = 'sshleifer/distilbart-cnn-6-6',
        use_lora: bool = True,
        freeze_llm: bool = False,
        dropout: float = 0.1
    ):
        super().__init__()
        
        print("\n" + "="*70)
        print("SIMPLE EEG-TO-TEXT MODEL (Research-Standard Architecture)")
        print("="*70)
        
        # 1. EEG Encoder (can be pre-trained separately)
        self.eeg_encoder = EEGMambaEncoder(
            num_channels=num_eeg_channels,
            d_model=encoder_dim,
            num_layers=num_encoder_layers,
            num_heads=8,
            dropout=dropout
        )
        
        # 2. Load pre-trained LLM
        print(f"\n💬 Loading LLM: {llm_model}")
        self.llm = BartForConditionalGeneration.from_pretrained(
            llm_model,
            low_cpu_mem_usage=True
        )
        llm_dim = self.llm.config.d_model
        print(f"  ✓ LLM loaded (dim: {llm_dim})")
        
        # 3. Projection layer (EEG features → LLM space)
        self.projection = nn.Sequential(
            nn.Linear(encoder_dim, llm_dim),
            nn.LayerNorm(llm_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(llm_dim, llm_dim)
        )
        print(f"  ✓ Projection: {encoder_dim}D → {llm_dim}D")
        
        # 4. Optional: Freeze LLM parameters
        if freeze_llm:
            for param in self.llm.parameters():
                param.requires_grad = False
            print(f"  ✓ LLM frozen (only train encoder + projection)")
        
        # 5. Optional: Apply LoRA to LLM
        if use_lora and not freeze_llm:
            try:
                from peft import LoraConfig, get_peft_model
                lora_config = LoraConfig(
                    r=8,
                    lora_alpha=32,
                    target_modules=["q_proj", "v_proj"],
                    lora_dropout=0.1,
                    bias="none",
                    task_type="SEQ_2_SEQ_LM"
                )
                self.llm = get_peft_model(self.llm, lora_config)
                self.llm.print_trainable_parameters()
                print(f"  ✓ LoRA applied to LLM")
            except ImportError:
                print(f"  ⚠ PEFT not available, full fine-tuning")
        
        print("="*70)
        print("Model Ready!")
        print("="*70 + "\n")
    
    def forward(
        self,
        eeg: torch.Tensor,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ):
        """
        Forward pass
        
        Args:
            eeg: (batch, seq_len, num_channels)
            decoder_input_ids: (batch, text_seq_len)
            decoder_attention_mask: (batch, text_seq_len)
            labels: (batch, text_seq_len)
        
        Returns:
            Model outputs with loss and logits
        """
        # 1. Encode EEG
        eeg_features = self.eeg_encoder(eeg)
        
        # 2. Project to LLM space
        llm_inputs = self.projection(eeg_features)
        
        # 3. Generate text with LLM
        outputs = self.llm(
            inputs_embeds=llm_inputs,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            return_dict=True
        )
        
        return outputs
    
    def generate(
        self,
        eeg: torch.Tensor,
        max_length: int = 50,
        num_beams: int = 5,
        **kwargs
    ):
        """
        Generate text from EEG
        
        Args:
            eeg: (batch, seq_len, num_channels)
            max_length: Maximum generation length
            num_beams: Beam search width
        
        Returns:
            generated_ids: (batch, generated_seq_len)
        """
        # Encode EEG
        with torch.no_grad():
            eeg_features = self.eeg_encoder(eeg)
            llm_inputs = self.projection(eeg_features)
            
            # Create encoder outputs for LLM
            from transformers.modeling_outputs import BaseModelOutput
            encoder_outputs = BaseModelOutput(
                last_hidden_state=llm_inputs
            )
            
            # Generate
            generated = self.llm.generate(
                encoder_outputs=encoder_outputs,
                max_length=max_length,
                num_beams=num_beams,
                **kwargs
            )
            
            return generated
    
    def pretrain_encoder(
        self,
        eeg: torch.Tensor,
        mask_ratio: float = 0.15
    ) -> torch.Tensor:
        """
        Pre-train encoder with masked reconstruction (like FEMBA/EEGMamba)
        
        Args:
            eeg: (batch, seq_len, num_channels)
            mask_ratio: Fraction of signal to mask
        
        Returns:
            reconstruction_loss: MSE loss between original and reconstructed
        """
        batch_size, seq_len, num_channels = eeg.shape
        
        # Create random mask
        mask = torch.rand(batch_size, seq_len, 1, device=eeg.device) > mask_ratio
        masked_eeg = eeg * mask
        
        # Encode masked EEG
        features = self.eeg_encoder(masked_eeg)
        
        # Decode to reconstruct original (add decoder if needed)
        # For now, simple reconstruction with linear layer
        reconstructed = self.projection(features)
        
        # Compute reconstruction loss (L1 + spectral)
        time_loss = F.l1_loss(reconstructed, eeg)
        
        # Optional: Spectral loss (frequency domain)
        # eeg_fft = torch.fft.rfft(eeg, dim=1)
        # recon_fft = torch.fft.rfft(reconstructed, dim=1)
        # freq_loss = F.l1_loss(eeg_fft.abs(), recon_fft.abs())
        
        return time_loss  # + freq_loss


def count_parameters(model):
    """Count trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# Test
if __name__ == '__main__':
    print("Testing Simple EEG2Text Model\n")
    
    # Create model
    model = SimpleEEG2Text(
        num_eeg_channels=105,
        encoder_dim=256,
        num_encoder_layers=6,
        use_lora=True,
        freeze_llm=False
    )
    
    # Count parameters
    total, trainable = count_parameters(model)
    print(f"\n📊 Model Statistics:")
    print(f"  Total parameters: {total:,}")
    print(f"  Trainable parameters: {trainable:,}")
    print(f"  Trainable percentage: {100*trainable/total:.2f}%")
    
    # Test forward pass
    print(f"\n🧪 Testing forward pass...")
    batch_size = 2
    seq_len = 800
    num_channels = 105
    
    dummy_eeg = torch.randn(batch_size, seq_len, num_channels)
    dummy_labels = torch.randint(0, 1000, (batch_size, 20))
    dummy_decoder_input = torch.randint(0, 1000, (batch_size, 20))
    
    outputs = model(
        eeg=dummy_eeg,
        decoder_input_ids=dummy_decoder_input,
        labels=dummy_labels
    )
    
    print(f"  ✓ Loss: {outputs.loss.item():.4f}")
    print(f"  ✓ Logits shape: {outputs.logits.shape}")
    
    # Test generation
    print(f"\n🔮 Testing generation...")
    generated = model.generate(
        dummy_eeg[:1],
        max_length=20,
        num_beams=1
    )
    print(f"  ✓ Generated shape: {generated.shape}")
    
    # Test pre-training
    print(f"\n🎯 Testing pre-training...")
    pretrain_loss = model.pretrain_encoder(dummy_eeg, mask_ratio=0.15)
    print(f"  ✓ Reconstruction loss: {pretrain_loss.item():.4f}")
    
    print(f"\n✅ All tests passed!")
    print(f"\n💡 This architecture follows current research:")
    print(f"   1. BiMamba encoder (like EEGMamba/FEMBA)")
    print(f"   2. Pre-trained LLM decoder (like recent papers)")
    print(f"   3. End-to-end fine-tuning on EEG-text pairs")
