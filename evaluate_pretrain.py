"""
Evaluate Pre-trained EEG Encoder with Visualizations and Metrics
Analyzes reconstruction quality, learned representations, and channel importance
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import json
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import scipy.stats as stats

from config import *
from models import EEGPreTraining
from dataset import load_preprocessed_data, create_dataloaders


class PretrainEvaluator:
    """Comprehensive evaluation of pre-trained EEG encoder"""
    
    def __init__(self, checkpoint_path, device='cpu'):
        self.device = torch.device(device)
        self.checkpoint_path = checkpoint_path
        
        # Load model
        print(f"Loading checkpoint from {checkpoint_path}")
        self.model = EEGPreTraining(
            num_channels=NUM_CHANNELS,
            sequence_length=MAX_EEG_LENGTH,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_TRANSFORMER_LAYERS,
            num_heads=NUM_HEADS,
            dropout=DROPOUT
        ).to(self.device)
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"✓ Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"✓ Training loss: {checkpoint.get('train_loss', 'N/A'):.4f}")
        print(f"✓ Validation loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
        
        # Create output directory
        self.output_dir = Path('results/pretrain_evaluation')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics = {}
        self.visualizations = []
    
    def evaluate_reconstruction(self, test_loader):
        """Evaluate reconstruction quality with detailed metrics"""
        print("\n📊 Evaluating Reconstruction Quality...")
        
        total_loss = 0
        total_mse = 0
        total_mae = 0
        total_correlation = []
        channel_errors = np.zeros(NUM_CHANNELS)
        temporal_errors = np.zeros(MAX_EEG_LENGTH)
        
        all_originals = []
        all_reconstructed = []
        all_masked_positions = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Reconstruction"):
                eeg = batch.to(self.device)
                
                # Get reconstruction
                loss, reconstructed, masked_eeg, mask = self.model(eeg, return_details=True)
                
                # Only evaluate masked regions
                mask_bool = mask.bool()
                original_masked = eeg[mask_bool]
                recon_masked = reconstructed[mask_bool]
                
                # Compute metrics
                total_loss += loss.item()
                mse = torch.nn.functional.mse_loss(recon_masked, original_masked)
                mae = torch.nn.functional.l1_loss(recon_masked, original_masked)
                total_mse += mse.item()
                total_mae += mae.item()
                
                # Correlation per sample
                for i in range(eeg.shape[0]):
                    orig = eeg[i][mask[i].bool()].cpu().numpy()
                    recon = reconstructed[i][mask[i].bool()].cpu().numpy()
                    if len(orig) > 1:
                        corr = np.corrcoef(orig.flatten(), recon.flatten())[0, 1]
                        if not np.isnan(corr):
                            total_correlation.append(corr)
                
                # Channel-wise error
                channel_mse = ((eeg - reconstructed) ** 2).mean(dim=(0, 2)).cpu().numpy()
                channel_errors += channel_mse
                
                # Temporal error
                temporal_mse = ((eeg - reconstructed) ** 2).mean(dim=(0, 1)).cpu().numpy()
                temporal_errors += temporal_mse
                
                # Store samples for visualization
                if len(all_originals) < 5:
                    all_originals.append(eeg[0].cpu().numpy())
                    all_reconstructed.append(reconstructed[0].cpu().numpy())
                    all_masked_positions.append(mask[0].cpu().numpy())
        
        num_batches = len(test_loader)
        channel_errors /= num_batches
        temporal_errors /= num_batches
        
        # Store metrics
        self.metrics['reconstruction'] = {
            'loss': total_loss / num_batches,
            'mse': total_mse / num_batches,
            'mae': total_mae / num_batches,
            'correlation_mean': np.mean(total_correlation),
            'correlation_std': np.std(total_correlation),
            'correlation_min': np.min(total_correlation),
            'correlation_max': np.max(total_correlation)
        }
        
        print(f"\n✓ Reconstruction Loss: {self.metrics['reconstruction']['loss']:.4f}")
        print(f"✓ MSE: {self.metrics['reconstruction']['mse']:.4f}")
        print(f"✓ MAE: {self.metrics['reconstruction']['mae']:.4f}")
        print(f"✓ Correlation: {self.metrics['reconstruction']['correlation_mean']:.4f} ± {self.metrics['reconstruction']['correlation_std']:.4f}")
        
        # Visualizations
        self._plot_reconstruction_samples(all_originals, all_reconstructed, all_masked_positions)
        self._plot_channel_errors(channel_errors)
        self._plot_temporal_errors(temporal_errors)
        self._plot_correlation_distribution(total_correlation)
        
        return self.metrics['reconstruction']
    
    def analyze_representations(self, test_loader, max_samples=1000):
        """Analyze learned representations using dimensionality reduction"""
        print("\n🧠 Analyzing Learned Representations...")
        
        all_embeddings = []
        all_subjects = []
        all_tasks = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Extracting embeddings"):
                eeg = batch.to(self.device)
                
                # Get embeddings from encoder
                embeddings = self.model.encoder(eeg)
                embeddings = embeddings.mean(dim=1)  # Global average pooling
                
                all_embeddings.append(embeddings.cpu().numpy())
                
                if len(all_embeddings) * eeg.shape[0] >= max_samples:
                    break
        
        embeddings = np.vstack(all_embeddings)[:max_samples]
        
        print(f"✓ Extracted {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")
        
        # PCA Analysis
        print("  - Running PCA...")
        pca = PCA(n_components=min(50, embeddings.shape[1]))
        pca_result = pca.fit_transform(embeddings)
        
        self.metrics['pca'] = {
            'explained_variance_ratio': pca.explained_variance_ratio_[:10].tolist(),
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_)[:10].tolist(),
            'total_variance_50d': float(np.sum(pca.explained_variance_ratio_))
        }
        
        print(f"  ✓ Top 10 components explain {self.metrics['pca']['cumulative_variance'][-1]:.2%} of variance")
        
        # t-SNE Analysis
        print("  - Running t-SNE (this may take a while)...")
        tsne = TSNE(n_components=2, random_state=SEED, perplexity=30)
        tsne_result = tsne.fit_transform(embeddings[:500])  # Limit for speed
        
        # Visualizations
        self._plot_pca_variance(pca)
        self._plot_embeddings_2d(pca_result[:, :2], "PCA")
        self._plot_embeddings_2d(tsne_result, "t-SNE")
        
        return embeddings, pca_result, tsne_result
    
    def analyze_attention_patterns(self, test_loader):
        """Analyze attention patterns in transformer layers"""
        print("\n👁️ Analyzing Attention Patterns...")
        
        attention_maps = []
        
        # Hook to capture attention weights
        def hook_fn(module, input, output):
            # output is (batch, num_heads, seq_len, seq_len)
            attention_maps.append(output[1].detach().cpu())
        
        # Register hooks on transformer layers
        hooks = []
        for layer in self.model.encoder.transformer_encoder.layers:
            hook = layer.self_attn.register_forward_hook(hook_fn)
            hooks.append(hook)
        
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(test_loader, desc="Attention")):
                if idx >= 5:  # Analyze first 5 batches
                    break
                eeg = batch.to(self.device)
                _ = self.model.encoder(eeg)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        if attention_maps:
            # Average attention across samples and heads
            avg_attention = torch.stack(attention_maps).mean(dim=0).mean(dim=0).numpy()
            self._plot_attention_heatmap(avg_attention)
        
        print("✓ Attention analysis complete")
    
    def evaluate_channel_importance(self, test_loader):
        """Evaluate importance of each EEG channel"""
        print("\n🔬 Evaluating Channel Importance...")
        
        baseline_loss = []
        channel_ablation_losses = np.zeros(NUM_CHANNELS)
        
        with torch.no_grad():
            # Get baseline loss
            for batch in tqdm(test_loader, desc="Baseline"):
                eeg = batch.to(self.device)
                loss = self.model(eeg)
                baseline_loss.append(loss.item())
            
            baseline_loss = np.mean(baseline_loss)
            
            # Ablate each channel
            for ch in tqdm(range(NUM_CHANNELS), desc="Channel ablation"):
                channel_losses = []
                
                for batch in test_loader:
                    eeg = batch.to(self.device).clone()
                    eeg[:, ch, :] = 0  # Zero out channel
                    
                    loss = self.model(eeg)
                    channel_losses.append(loss.item())
                
                channel_ablation_losses[ch] = np.mean(channel_losses)
        
        # Channel importance = increase in loss when ablated
        channel_importance = channel_ablation_losses - baseline_loss
        
        self.metrics['channel_importance'] = {
            'baseline_loss': float(baseline_loss),
            'importance_scores': channel_importance.tolist(),
            'top_10_channels': np.argsort(channel_importance)[-10:].tolist(),
            'bottom_10_channels': np.argsort(channel_importance)[:10].tolist()
        }
        
        print(f"✓ Baseline loss: {baseline_loss:.4f}")
        print(f"✓ Most important channels: {self.metrics['channel_importance']['top_10_channels']}")
        
        self._plot_channel_importance(channel_importance)
        
        return channel_importance
    
    def _plot_reconstruction_samples(self, originals, reconstructed, masks):
        """Plot sample reconstructions"""
        fig, axes = plt.subplots(len(originals), 3, figsize=(15, 3*len(originals)))
        
        for i, (orig, recon, mask) in enumerate(zip(originals, reconstructed, masks)):
            # Select first 5 channels for visualization
            channels_to_plot = min(5, orig.shape[0])
            
            ax1, ax2, ax3 = axes[i] if len(originals) > 1 else axes
            
            # Original
            ax1.imshow(orig[:channels_to_plot, :500], aspect='auto', cmap='viridis')
            ax1.set_title(f'Sample {i+1}: Original')
            ax1.set_ylabel('Channel')
            ax1.set_xlabel('Time')
            
            # Reconstructed
            ax2.imshow(recon[:channels_to_plot, :500], aspect='auto', cmap='viridis')
            ax2.set_title('Reconstructed')
            ax2.set_xlabel('Time')
            
            # Difference
            diff = np.abs(orig - recon)
            ax3.imshow(diff[:channels_to_plot, :500], aspect='auto', cmap='hot')
            ax3.set_title('Absolute Error')
            ax3.set_xlabel('Time')
        
        plt.tight_layout()
        save_path = self.output_dir / 'reconstruction_samples.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {save_path}")
    
    def _plot_channel_errors(self, channel_errors):
        """Plot per-channel reconstruction errors"""
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(channel_errors)), channel_errors)
        plt.xlabel('Channel Index')
        plt.ylabel('Mean Squared Error')
        plt.title('Reconstruction Error by Channel')
        plt.grid(True, alpha=0.3)
        
        save_path = self.output_dir / 'channel_errors.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {save_path}")
    
    def _plot_temporal_errors(self, temporal_errors):
        """Plot temporal reconstruction errors"""
        plt.figure(figsize=(12, 6))
        plt.plot(temporal_errors)
        plt.xlabel('Time Step')
        plt.ylabel('Mean Squared Error')
        plt.title('Reconstruction Error Over Time')
        plt.grid(True, alpha=0.3)
        
        save_path = self.output_dir / 'temporal_errors.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {save_path}")
    
    def _plot_correlation_distribution(self, correlations):
        """Plot distribution of reconstruction correlations"""
        plt.figure(figsize=(10, 6))
        plt.hist(correlations, bins=50, edgecolor='black', alpha=0.7)
        plt.axvline(np.mean(correlations), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(correlations):.3f}')
        plt.xlabel('Correlation Coefficient')
        plt.ylabel('Frequency')
        plt.title('Distribution of Reconstruction Correlations')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        save_path = self.output_dir / 'correlation_distribution.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {save_path}")
    
    def _plot_pca_variance(self, pca):
        """Plot PCA explained variance"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Individual variance
        ax1.bar(range(1, 11), pca.explained_variance_ratio_[:10])
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Explained Variance Ratio')
        ax1.set_title('Explained Variance by Component')
        ax1.grid(True, alpha=0.3)
        
        # Cumulative variance
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        ax2.plot(range(1, len(cumsum)+1), cumsum)
        ax2.axhline(0.95, color='red', linestyle='--', label='95% threshold')
        ax2.set_xlabel('Number of Components')
        ax2.set_ylabel('Cumulative Explained Variance')
        ax2.set_title('Cumulative Explained Variance')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / 'pca_variance.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {save_path}")
    
    def _plot_embeddings_2d(self, embeddings_2d, method_name):
        """Plot 2D embeddings"""
        plt.figure(figsize=(10, 8))
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                   alpha=0.5, s=20, cmap='viridis')
        plt.xlabel(f'{method_name} Component 1')
        plt.ylabel(f'{method_name} Component 2')
        plt.title(f'EEG Embeddings Visualization ({method_name})')
        plt.colorbar(label='Sample Index')
        plt.grid(True, alpha=0.3)
        
        save_path = self.output_dir / f'embeddings_{method_name.lower()}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {save_path}")
    
    def _plot_attention_heatmap(self, attention):
        """Plot attention heatmap"""
        plt.figure(figsize=(12, 10))
        sns.heatmap(attention[:100, :100], cmap='viridis', square=True)
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')
        plt.title('Average Attention Pattern (First 100 timesteps)')
        
        save_path = self.output_dir / 'attention_heatmap.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {save_path}")
    
    def _plot_channel_importance(self, importance):
        """Plot channel importance scores"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Bar plot
        ax1.bar(range(len(importance)), importance)
        ax1.set_xlabel('Channel Index')
        ax1.set_ylabel('Importance Score (Δ Loss)')
        ax1.set_title('Channel Importance Scores')
        ax1.grid(True, alpha=0.3)
        
        # Top channels
        top_indices = np.argsort(importance)[-20:]
        top_scores = importance[top_indices]
        ax2.barh(range(len(top_scores)), top_scores)
        ax2.set_yticks(range(len(top_scores)))
        ax2.set_yticklabels([f'Ch {i}' for i in top_indices])
        ax2.set_xlabel('Importance Score')
        ax2.set_title('Top 20 Most Important Channels')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / 'channel_importance.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {save_path}")
    
    def save_results(self):
        """Save all metrics to JSON"""
        output_file = self.output_dir / 'pretrain_metrics.json'
        
        with open(output_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        print(f"\n💾 Saved metrics to: {output_file}")
        
        # Create summary report
        report_file = self.output_dir / 'evaluation_report.txt'
        with open(report_file, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("Pre-trained Model Evaluation Report\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Checkpoint: {self.checkpoint_path}\n\n")
            
            if 'reconstruction' in self.metrics:
                f.write("Reconstruction Quality:\n")
                f.write(f"  Loss: {self.metrics['reconstruction']['loss']:.4f}\n")
                f.write(f"  MSE: {self.metrics['reconstruction']['mse']:.4f}\n")
                f.write(f"  MAE: {self.metrics['reconstruction']['mae']:.4f}\n")
                f.write(f"  Correlation: {self.metrics['reconstruction']['correlation_mean']:.4f} ")
                f.write(f"± {self.metrics['reconstruction']['correlation_std']:.4f}\n\n")
            
            if 'pca' in self.metrics:
                f.write("Representation Analysis:\n")
                f.write(f"  PCA variance (50 components): {self.metrics['pca']['total_variance_50d']:.2%}\n")
                f.write(f"  Top 10 PCs variance: {self.metrics['pca']['cumulative_variance'][-1]:.2%}\n\n")
            
            if 'channel_importance' in self.metrics:
                f.write("Channel Importance:\n")
                f.write(f"  Baseline loss: {self.metrics['channel_importance']['baseline_loss']:.4f}\n")
                f.write(f"  Most important channels: {self.metrics['channel_importance']['top_10_channels']}\n\n")
            
            f.write("\nAll visualizations saved to: results/pretrain_evaluation/\n")
        
        print(f"📄 Saved report to: {report_file}")


def main():
    """Run complete pre-training evaluation"""
    print("=" * 70)
    print("EEG Pre-training Model Evaluation")
    print("=" * 70)
    
    # Configuration
    checkpoint_path = 'models/pretraining/best_pretrain.pt'
    
    if not Path(checkpoint_path).exists():
        print(f"\n❌ Checkpoint not found: {checkpoint_path}")
        print("Please run pretraining first: python pretrain.py")
        return
        
    # Load data
    print("\n📂 Loading test data...")
    all_data = load_preprocessed_data()
    _, _, test_data = all_data  # Get test split
    
    _, _, test_loader = create_dataloaders(
        all_data, 
        batch_size=4,  # Small batch for evaluation
        for_pretraining=True
    )
    
    print(f"✓ Loaded {len(test_data)} test samples")
    
    # Initialize evaluator
    evaluator = PretrainEvaluator(checkpoint_path, device='cpu')
    
    # Run evaluations
    print("\n" + "=" * 70)
    
    # 1. Reconstruction quality
    evaluator.evaluate_reconstruction(test_loader)
    
    # 2. Representation analysis
    evaluator.analyze_representations(test_loader, max_samples=1000)
    
    # 3. Channel importance (this takes longer)
    print("\n⚠️ Channel ablation may take several minutes...")
    # evaluator.evaluate_channel_importance(test_loader)  # Uncomment for full analysis
    print("  (Skipped for speed - uncomment to run full analysis)")
    
    # Save results
    print("\n" + "=" * 70)
    evaluator.save_results()
    
    print("\n" + "=" * 70)
    print("✅ Evaluation Complete!")
    print("=" * 70)
    print(f"\nResults saved to: {evaluator.output_dir}")
    print("\nGenerated files:")
    print("  - pretrain_metrics.json: All metrics")
    print("  - evaluation_report.txt: Summary report")
    print("  - *.png: Visualizations")


if __name__ == '__main__':
    main()
