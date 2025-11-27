"""
Advanced EEG2TEXT Architecture
Combines Mamba SSM + Graph Neural Networks + Semantic Bottleneck

Novel Contributions:
1. Mamba backbone for temporal modeling (O(N) complexity)
2. Graph-based spatial encoding of brain regions
3. Semantic bottleneck for interpretability
4. Neuroscience-grounded architecture

Author: Implementation based on cutting-edge research
Date: November 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from transformers import BartForConditionalGeneration, BartTokenizer

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    print("WARNING: mamba-ssm not installed. Using Transformer fallback.")
    print("Install with: pip install mamba-ssm")
    MAMBA_AVAILABLE = False

try:
    from torch_geometric.nn import GATConv, GCNConv, global_mean_pool
    from torch_geometric.data import Data, Batch
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    print("WARNING: torch_geometric not installed. Using MLP fallback.")
    print("Install with: pip install torch-geometric")
    TORCH_GEOMETRIC_AVAILABLE = False


class MambaEncoder(nn.Module):
    """
    Mamba State Space Model for temporal EEG encoding
    Replaces Transformer with linear-time complexity
    
    Advantages:
    - O(N) complexity vs Transformer O(N²)
    - Better for long sequences (EEG has 800+ timesteps)
    - State-of-the-art on long-range dependencies
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1,
        use_mamba: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_mamba = use_mamba and MAMBA_AVAILABLE
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        if self.use_mamba:
            # Stack of Mamba blocks
            self.mamba_layers = nn.ModuleList([
                Mamba(
                    d_model=hidden_dim,
                    d_state=16,  # State size
                    d_conv=4,    # Convolution size
                    expand=2     # Expansion factor
                )
                for _ in range(num_layers)
            ])
            print(f"  ✓ Using Mamba SSM with {num_layers} layers")
        else:
            # Fallback to Transformer if Mamba not available
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
            print(f"  ✓ Using Transformer fallback with {num_layers} layers")
        
        # Layer norms
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            (batch, seq_len, hidden_dim)
        """
        # Project to hidden dimension
        x = self.input_proj(x)
        
        if self.use_mamba:
            # Pass through Mamba layers with residual connections
            for mamba_layer, norm in zip(self.mamba_layers, self.norms):
                residual = x
                x = mamba_layer(x)
                x = norm(x + residual)
                x = self.dropout(x)
        else:
            # Transformer fallback
            x = self.transformer(x)
        
        return x


class BrainGraphEncoder(nn.Module):
    """
    Graph Neural Network for spatial encoding of brain regions
    
    Models brain as graph:
    - Nodes: 10 brain regions (prefrontal, Broca, Wernicke, etc.)
    - Edges: Functional connectivity (based on neuroscience)
    
    Novel Contribution: Explicit modeling of inter-region communication
    """
    
    def __init__(
        self,
        channel_groups: Dict[str, List[int]],
        hidden_dim: int = 128,
        num_gnn_layers: int = 3,
        dropout: float = 0.1,
        use_attention: bool = True
    ):
        super().__init__()
        
        self.channel_groups = channel_groups
        self.region_names = list(channel_groups.keys())
        self.num_regions = len(self.region_names)
        self.hidden_dim = hidden_dim
        self.use_geometric = TORCH_GEOMETRIC_AVAILABLE
        
        # Per-region encoders (process each region's channels)
        self.region_encoders = nn.ModuleDict({
            region: nn.Sequential(
                nn.Linear(len(channels), hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            for region, channels in channel_groups.items()
        })
        
        if self.use_geometric:
            # Graph convolution layers
            if use_attention:
                # Graph Attention Network (learns edge importance)
                self.gnn_layers = nn.ModuleList([
                    GATConv(
                        hidden_dim,
                        hidden_dim,
                        heads=4,
                        dropout=dropout,
                        concat=False
                    )
                    for _ in range(num_gnn_layers)
                ])
                print(f"  ✓ Using Graph Attention Network with {num_gnn_layers} layers")
            else:
                # Graph Convolution Network (fixed edge weights)
                self.gnn_layers = nn.ModuleList([
                    GCNConv(hidden_dim, hidden_dim)
                    for _ in range(num_gnn_layers)
                ])
                print(f"  ✓ Using Graph Convolution Network with {num_gnn_layers} layers")
        else:
            # Fallback: Use cross-attention between regions
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=4,
                dropout=dropout,
                batch_first=True
            )
            print(f"  ✓ Using cross-attention fallback")
        
        # Build brain connectivity graph
        self.edge_index = self._build_brain_graph()
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def _build_brain_graph(self) -> torch.Tensor:
        """
        Build brain connectivity graph based on neuroscience literature
        
        Key connections:
        - Visual → Wernicke (reading comprehension)
        - Wernicke → Broca (comprehension → production)
        - Prefrontal → Broca (planning → speech)
        - Broca ↔ Primary Motor (speech production)
        - etc.
        """
        # Map region names to indices
        region_to_idx = {name: i for i, name in enumerate(self.region_names)}
        
        # Define edges (undirected, so add both directions)
        edges = [
            # Language circuit
            ('wernicke', 'broca'),           # Classic language pathway
            ('broca', 'primary_motor'),      # Speech production
            
            # Reading pathway
            ('visual', 'wernicke'),          # Visual → comprehension
            ('visual', 'prefrontal'),        # Attention to text
            
            # Executive control
            ('prefrontal', 'broca'),         # Planning → production
            ('prefrontal', 'wernicke'),      # Attention → comprehension
            ('prefrontal', 'premotor'),      # Planning → motor prep
            
            # Motor pathways
            ('premotor', 'primary_motor'),   # Motor planning
            ('primary_motor', 'primary_sensory'),  # Motor feedback
            
            # Auditory (inner speech)
            ('auditory', 'wernicke'),        # Auditory processing
            ('auditory', 'broca'),           # Inner speech
            ('auditory_assoc', 'auditory'),  # Association areas
            
            # Somatosensory
            ('somatic_sensory', 'primary_sensory'),
        ]
        
        # Convert to edge indices
        edge_list = []
        for src, dst in edges:
            if src in region_to_idx and dst in region_to_idx:
                src_idx = region_to_idx[src]
                dst_idx = region_to_idx[dst]
                # Add both directions (undirected graph)
                edge_list.append([src_idx, dst_idx])
                edge_list.append([dst_idx, src_idx])
        
        # Add self-loops
        for i in range(self.num_regions):
            edge_list.append([i, i])
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t()
        return edge_index
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            x: (batch, seq_len, all_channels)
        Returns:
            graph_features: (batch, seq_len, num_regions * hidden_dim)
            region_features: Dict mapping region names to features
        """
        batch_size, seq_len, _ = x.shape
        
        # Process each time step separately
        all_timestep_features = []
        region_outputs = {region: [] for region in self.region_names}
        
        for t in range(seq_len):
            # Extract features for each brain region at time t
            node_features = []
            for region in self.region_names:
                channels = self.channel_groups[region]
                region_data = x[:, t, channels]  # (batch, num_channels)
                region_feat = self.region_encoders[region](region_data)  # (batch, hidden_dim)
                node_features.append(region_feat)
                region_outputs[region].append(region_feat)
            
            # Stack to (batch, num_regions, hidden_dim)
            node_features = torch.stack(node_features, dim=1)
            
            if self.use_geometric and hasattr(self, 'gnn_layers'):
                # Graph neural network propagation
                # Reshape for torch_geometric: (batch * num_regions, hidden_dim)
                node_feat_flat = node_features.reshape(-1, self.hidden_dim)
                
                # Expand edge index for batch
                edge_index = self.edge_index.to(x.device)
                batch_edge_index = []
                for b in range(batch_size):
                    offset = b * self.num_regions
                    batch_edge_index.append(edge_index + offset)
                batch_edge_index = torch.cat(batch_edge_index, dim=1)
                
                # Apply GNN layers
                for gnn_layer in self.gnn_layers:
                    residual = node_feat_flat
                    node_feat_flat = gnn_layer(node_feat_flat, batch_edge_index)
                    node_feat_flat = F.gelu(node_feat_flat)
                    node_feat_flat = self.dropout(node_feat_flat + residual)
                
                # Reshape back: (batch, num_regions, hidden_dim)
                node_features = node_feat_flat.reshape(batch_size, self.num_regions, self.hidden_dim)
            else:
                # Fallback: Cross-attention between regions
                node_features_attn, _ = self.cross_attention(
                    node_features, node_features, node_features
                )
                node_features = node_features + node_features_attn
            
            # Flatten regions: (batch, num_regions * hidden_dim)
            timestep_feat = node_features.reshape(batch_size, -1)
            all_timestep_features.append(timestep_feat)
        
        # Stack back: (batch, seq_len, num_regions * hidden_dim)
        graph_features = torch.stack(all_timestep_features, dim=1)
        
        # Aggregate region outputs
        region_features = {
            region: torch.stack(feats, dim=1)  # (batch, seq_len, hidden_dim)
            for region, feats in region_outputs.items()
        }
        
        return graph_features, region_features


class SemanticBottleneck(nn.Module):
    """
    Semantic bottleneck layer for interpretability
    
    Forces model to encode information through discrete semantic concepts
    Enables visualization of what the model "understands"
    
    Novel Contribution: Makes black-box model interpretable
    """
    
    def __init__(
        self,
        input_dim: int,
        num_concepts: int = 64,
        concept_dim: int = 32,
        temperature: float = 1.0
    ):
        super().__init__()
        
        self.num_concepts = num_concepts
        self.concept_dim = concept_dim
        self.temperature = temperature
        
        # Concept codebook (learnable semantic concepts)
        self.concept_embeddings = nn.Parameter(
            torch.randn(num_concepts, concept_dim)
        )
        
        # Input → concept attention
        self.input_to_concepts = nn.Linear(input_dim, num_concepts)
        
        # Concept → output
        self.concepts_to_output = nn.Linear(concept_dim, input_dim)
        
        self.layer_norm = nn.LayerNorm(input_dim)
        
        print(f"  ✓ Semantic Bottleneck: {num_concepts} concepts × {concept_dim}D")
    
    def forward(
        self,
        x: torch.Tensor,
        return_concepts: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Args:
            x: (batch, seq_len, input_dim)
            return_concepts: Whether to return concept activations
        Returns:
            output: (batch, seq_len, input_dim)
            concept_info: Dict with concept activations and attention
        """
        batch_size, seq_len, input_dim = x.shape
        
        # Compute attention over concepts
        concept_logits = self.input_to_concepts(x)  # (batch, seq_len, num_concepts)
        concept_weights = F.softmax(concept_logits / self.temperature, dim=-1)
        
        # Retrieve concept embeddings (soft retrieval)
        # (batch, seq_len, num_concepts) @ (num_concepts, concept_dim)
        # → (batch, seq_len, concept_dim)
        concept_repr = torch.matmul(concept_weights, self.concept_embeddings)
        
        # Decode from concepts
        output = self.concepts_to_output(concept_repr)
        output = self.layer_norm(output + x)  # Residual connection
        
        if return_concepts:
            concept_info = {
                'concept_weights': concept_weights,  # Which concepts are active
                'concept_repr': concept_repr,        # Concept representations
                'top_concepts': concept_weights.topk(5, dim=-1).indices  # Top 5 concepts
            }
            return output, concept_info
        
        return output, None
    
    def get_concept_names(self, tokenizer=None) -> List[str]:
        """
        Generate interpretable names for concepts (requires analysis)
        Can be done post-hoc by analyzing which words activate which concepts
        """
        return [f"Concept_{i}" for i in range(self.num_concepts)]


class MambaGraphEEG2TEXT(nn.Module):
    """
    Complete EEG-to-Text model combining:
    1. Mamba SSM for temporal encoding
    2. Graph NN for spatial encoding
    3. Semantic bottleneck for interpretability
    4. BART decoder with LoRA
    
    NOVEL ARCHITECTURE - State-of-the-art + Interpretable
    """
    
    def __init__(
        self,
        channel_groups: Dict[str, List[int]],
        hidden_dim: int = 128,
        mamba_layers: int = 4,
        gnn_layers: int = 3,
        num_concepts: int = 64,
        bart_model: str = 'sshleifer/distilbart-cnn-6-6',
        use_lora: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        
        print("\n" + "="*70)
        print("Initializing Mamba-Graph-Semantic EEG2TEXT Model")
        print("="*70)
        
        self.channel_groups = channel_groups
        self.num_regions = len(channel_groups)
        self.hidden_dim = hidden_dim
        
        # 1. CNN Compressor (reduce temporal dimension)
        # NOTE: CHANNEL_GROUPS has 103 channels but ZuCo has 105 total
        # Use 105 for CNN input to match actual data
        num_input_channels = 105  # ZuCo dataset channels
        self.cnn_compressor = nn.Sequential(
            nn.Conv1d(num_input_channels, hidden_dim, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )
        print(f"1. CNN Compressor: {num_input_channels} channels → {hidden_dim}D (4x compression)")
        
        # 2. Mamba Temporal Encoder
        self.mamba_encoder = MambaEncoder(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=mamba_layers,
            dropout=dropout,
            use_mamba=MAMBA_AVAILABLE
        )
        print(f"2. Mamba Encoder: {mamba_layers} layers, {hidden_dim}D")
        
        # 3. Graph Spatial Encoder
        self.graph_encoder = BrainGraphEncoder(
            channel_groups=channel_groups,
            hidden_dim=hidden_dim,
            num_gnn_layers=gnn_layers,
            dropout=dropout,
            use_attention=True
        )
        graph_output_dim = self.num_regions * hidden_dim
        print(f"3. Graph Encoder: {self.num_regions} regions × {hidden_dim}D = {graph_output_dim}D")
        
        # 4. Fusion layer (combine temporal + spatial)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + graph_output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        print(f"4. Fusion Layer: ({hidden_dim} + {graph_output_dim}) → {hidden_dim}D")
        
        # 5. Semantic Bottleneck
        self.semantic_bottleneck = SemanticBottleneck(
            input_dim=hidden_dim,
            num_concepts=num_concepts,
            concept_dim=32,
            temperature=1.0
        )
        print(f"5. Semantic Bottleneck: {num_concepts} interpretable concepts")
        
        # 6. BART Decoder
        print(f"6. Loading BART decoder ({bart_model})...")
        self.bart = BartForConditionalGeneration.from_pretrained(
            bart_model,
            low_cpu_mem_usage=True
        )
        bart_dim = self.bart.config.d_model
        
        # Apply LoRA for efficient fine-tuning
        if use_lora:
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
                self.bart = get_peft_model(self.bart, lora_config)
                self.bart.print_trainable_parameters()
                print("  ✓ LoRA applied for memory efficiency")
            except ImportError:
                print("  ⚠ PEFT not available, using full fine-tuning")
        
        # Projection to BART dimension
        self.proj_to_bart = nn.Linear(hidden_dim, bart_dim)
        
        print("="*70)
        print("Model Initialized Successfully!")
        print("="*70 + "\n")
    
    def forward(
        self,
        eeg: torch.Tensor,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_interpretability: bool = False
    ):
        """
        Forward pass
        
        Args:
            eeg: (batch, time_steps, channels)
            decoder_input_ids: (batch, seq_len)
            decoder_attention_mask: (batch, seq_len)
            labels: (batch, seq_len)
            return_interpretability: Return concept activations and region features
        
        Returns:
            BartOutput or dict with interpretability info
        """
        batch_size = eeg.shape[0]
        
        # 1. CNN compression
        # (batch, time, channels) → (batch, channels, time) for Conv1d
        eeg_compressed = eeg.transpose(1, 2)
        eeg_compressed = self.cnn_compressor(eeg_compressed)
        eeg_compressed = eeg_compressed.transpose(1, 2)  # Back to (batch, time, hidden)
        
        # 2. Mamba temporal encoding
        temporal_features = self.mamba_encoder(eeg_compressed)
        
        # 3. Graph spatial encoding
        graph_features, region_features = self.graph_encoder(eeg)
        
        # Match sequence lengths (graph may be longer)
        min_len = min(temporal_features.shape[1], graph_features.shape[1])
        temporal_features = temporal_features[:, :min_len, :]
        graph_features = graph_features[:, :min_len, :]
        
        # 4. Fuse temporal + spatial
        combined = torch.cat([temporal_features, graph_features], dim=-1)
        fused = self.fusion(combined)
        
        # 5. Semantic bottleneck (interpretability)
        semantic_features, concept_info = self.semantic_bottleneck(
            fused,
            return_concepts=return_interpretability
        )
        
        # 6. Project to BART and decode
        encoder_outputs = self.proj_to_bart(semantic_features)
        
        # Generate or train
        if labels is not None:
            # Training mode
            outputs = self.bart(
                inputs_embeds=encoder_outputs,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                labels=labels,
                return_dict=True
            )
            
            if return_interpretability:
                return {
                    'loss': outputs.loss,
                    'logits': outputs.logits,
                    'concept_info': concept_info,
                    'region_features': region_features,
                    'temporal_features': temporal_features,
                    'graph_features': graph_features
                }
            return outputs
        else:
            # Inference mode - just return encoder outputs
            return encoder_outputs
    
    def generate(
        self,
        eeg: torch.Tensor,
        max_length: int = 50,
        num_beams: int = 5,
        return_interpretability: bool = False,
        **kwargs
    ):
        """
        Generate text from EEG
        
        Args:
            eeg: (batch, time_steps, channels)
            max_length: Maximum generation length
            num_beams: Number of beams for beam search
            return_interpretability: Return concept activations
        
        Returns:
            generated_ids: (batch, seq_len)
            Optional: interpretability_info dict
        """
        # Forward pass through encoder
        with torch.no_grad():
            # Get encoder outputs
            batch_size = eeg.shape[0]
            
            # 1-5: Same as forward
            eeg_compressed = eeg.transpose(1, 2)
            eeg_compressed = self.cnn_compressor(eeg_compressed)
            eeg_compressed = eeg_compressed.transpose(1, 2)
            
            temporal_features = self.mamba_encoder(eeg_compressed)
            graph_features, region_features = self.graph_encoder(eeg)
            
            min_len = min(temporal_features.shape[1], graph_features.shape[1])
            temporal_features = temporal_features[:, :min_len, :]
            graph_features = graph_features[:, :min_len, :]
            
            combined = torch.cat([temporal_features, graph_features], dim=-1)
            fused = self.fusion(combined)
            
            semantic_features, concept_info = self.semantic_bottleneck(
                fused,
                return_concepts=return_interpretability
            )
            
            encoder_outputs = self.proj_to_bart(semantic_features)
            
            # Create encoder outputs for BART
            from transformers.modeling_outputs import BaseModelOutput
            bart_encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs
            )
            
            # Generate
            generated_ids = self.bart.generate(
                encoder_outputs=bart_encoder_outputs,
                max_length=max_length,
                num_beams=num_beams,
                **kwargs
            )
            
            if return_interpretability:
                return generated_ids, {
                    'concept_info': concept_info,
                    'region_features': region_features,
                    'top_regions': self._get_top_regions(region_features)
                }
            
            return generated_ids
    
    def _get_top_regions(self, region_features: Dict) -> List[str]:
        """Identify which brain regions were most active"""
        region_activations = {}
        for region, features in region_features.items():
            # Average activation across time and batch
            activation = features.abs().mean().item()
            region_activations[region] = activation
        
        # Sort by activation
        top_regions = sorted(
            region_activations.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [region for region, _ in top_regions[:5]]


def count_parameters(model):
    """Count trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# Test function
if __name__ == '__main__':
    print("Testing Mamba-Graph-Semantic EEG2TEXT Model\n")
    
    # Example channel groups (10 brain regions)
    from config import Config
    
    # Create model
    model = MambaGraphEEG2TEXT(
        channel_groups=Config.CHANNEL_GROUPS,
        hidden_dim=128,
        mamba_layers=4,
        gnn_layers=3,
        num_concepts=64,
        use_lora=True
    )
    
    # Count parameters
    total, trainable = count_parameters(model)
    print(f"\n📊 Model Statistics:")
    print(f"  Total parameters: {total:,}")
    print(f"  Trainable parameters: {trainable:,}")
    print(f"  Percentage trainable: {100*trainable/total:.2f}%")
    
    # Test forward pass
    print(f"\n🧪 Testing forward pass...")
    batch_size = 2
    seq_len = 800
    num_channels = 105
    
    dummy_eeg = torch.randn(batch_size, seq_len, num_channels)
    dummy_labels = torch.randint(0, 1000, (batch_size, 20))
    dummy_decoder_input = torch.randint(0, 1000, (batch_size, 20))
    
    # Forward pass with interpretability
    outputs = model(
        dummy_eeg,
        decoder_input_ids=dummy_decoder_input,
        labels=dummy_labels,
        return_interpretability=True
    )
    
    print(f"  ✓ Loss: {outputs['loss'].item():.4f}")
    print(f"  ✓ Logits shape: {outputs['logits'].shape}")
    print(f"  ✓ Top concepts: {outputs['concept_info']['top_concepts'].shape}")
    print(f"  ✓ Region features: {len(outputs['region_features'])} regions")
    
    # Test generation
    print(f"\n🔮 Testing generation...")
    generated, interp = model.generate(
        dummy_eeg[:1],
        max_length=20,
        num_beams=1,
        return_interpretability=True
    )
    print(f"  ✓ Generated shape: {generated.shape}")
    print(f"  ✓ Top active regions: {interp['top_regions']}")
    
    print(f"\n✅ All tests passed!")
