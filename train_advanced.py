"""
Training script for Advanced Mamba-Graph-Semantic EEG2TEXT Model

Includes:
- Interpretability logging (concept activations, region analysis)
- Advanced optimizations
- Curriculum learning
- Semantic consistency loss
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU (remove if using GPU)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BartTokenizer
from tqdm import tqdm
import time
import json
from pathlib import Path
import numpy as np

from config import Config
from models_advanced import MambaGraphEEG2TEXT, count_parameters
from dataset import load_preprocessed_data, split_data, create_dataloaders
from utils import (
    set_seed, AverageMeter, save_checkpoint, load_checkpoint,
    get_device_info, WarmupLinearSchedule, format_time
)


class SemanticConsistencyLoss(nn.Module):
    """
    Additional loss to encourage semantic similarity
    Uses sentence embeddings to ensure generated text is semantically close
    """
    def __init__(self):
        super().__init__()
        try:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            self.available = True
            print("  ✓ Semantic consistency loss enabled")
        except ImportError:
            self.available = False
            print("  ⚠ sentence-transformers not available, skipping semantic loss")
    
    def forward(self, generated_text, reference_text):
        if not self.available:
            return torch.tensor(0.0)
        
        # Encode sentences
        gen_emb = self.encoder.encode(generated_text, convert_to_tensor=True)
        ref_emb = self.encoder.encode(reference_text, convert_to_tensor=True)
        
        # Cosine similarity loss
        similarity = F.cosine_similarity(gen_emb, ref_emb, dim=-1)
        return 1.0 - similarity.mean()


class AdvancedTrainer:
    """
    Trainer for Mamba-Graph-Semantic model with interpretability logging
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        device,
        learning_rate=1e-5,
        warmup_steps=500,
        weight_decay=0.01,
        gradient_clip=0.5,
        semantic_loss_weight=0.0,  # Set > 0 to enable
        log_interpretability=True
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.gradient_clip = gradient_clip
        self.log_interpretability = log_interpretability
        self.semantic_loss_weight = semantic_loss_weight
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Scheduler
        self.scheduler = WarmupLinearSchedule(
            self.optimizer,
            warmup_steps=warmup_steps,
            total_steps=100000  # Will be updated
        )
        
        # Optional semantic loss
        if semantic_loss_weight > 0:
            self.semantic_loss = SemanticConsistencyLoss()
        else:
            self.semantic_loss = None
        
        # Tracking
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Interpretability logs
        self.concept_logs = []
        self.region_logs = []
    
    def train_epoch(
        self,
        train_loader,
        epoch,
        accumulation_steps=8
    ):
        """Train for one epoch"""
        self.model.train()
        
        losses = AverageMeter()
        ce_losses = AverageMeter()
        semantic_losses = AverageMeter()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            eeg = batch['eeg'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass with interpretability every N steps
            return_interp = (
                self.log_interpretability and
                self.global_step % 100 == 0
            )
            
            outputs = self.model(
                eeg=eeg,
                decoder_input_ids=input_ids,
                decoder_attention_mask=attention_mask,
                labels=labels,
                return_interpretability=return_interp
            )
            
            # Get losses
            if isinstance(outputs, dict):
                ce_loss = outputs['loss']
            else:
                ce_loss = outputs.loss
            
            loss = ce_loss
            
            # Optional semantic loss
            if self.semantic_loss is not None and batch_idx % 10 == 0:
                # Generate sample for semantic loss (expensive, do sparingly)
                with torch.no_grad():
                    generated = self.model.generate(
                        eeg[:1],
                        max_length=30,
                        num_beams=1
                    )
                    gen_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
                    ref_text = batch['text'][0] if 'text' in batch else ""
                
                if ref_text:
                    sem_loss = self.semantic_loss(gen_text, ref_text)
                    loss = loss + self.semantic_loss_weight * sem_loss
                    semantic_losses.update(sem_loss.item())
            
            # Backward with gradient accumulation
            loss = loss / accumulation_steps
            loss.backward()
            
            # Update weights
            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                self.global_step += 1
            
            # Log interpretability
            if return_interp and isinstance(outputs, dict):
                self._log_interpretability(outputs, batch)
            
            # Update metrics
            losses.update(loss.item() * accumulation_steps)
            ce_losses.update(ce_loss.item())
            
            # Progress bar
            progress_bar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })
            
            # NaN detection
            if torch.isnan(loss):
                print(f"\n❌ NaN detected at step {self.global_step}!")
                return None
        
        return {
            'loss': losses.avg,
            'ce_loss': ce_losses.avg,
            'semantic_loss': semantic_losses.avg if semantic_losses.count > 0 else 0
        }
    
    def validate(self, val_loader):
        """Validation"""
        self.model.eval()
        losses = AverageMeter()
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                eeg = batch['eeg'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    eeg=eeg,
                    decoder_input_ids=input_ids,
                    decoder_attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs['loss']
                losses.update(loss.item())
        
        return losses.avg
    
    def _log_interpretability(self, outputs, batch):
        """Log concept activations and region importance"""
        if 'concept_info' not in outputs:
            return
        
        concept_info = outputs['concept_info']
        region_features = outputs.get('region_features', {})
        
        # Log top concepts
        top_concepts = concept_info['top_concepts'][0].cpu().numpy()  # First sample
        concept_weights = concept_info['concept_weights'][0].cpu().numpy()
        
        self.concept_logs.append({
            'step': self.global_step,
            'top_concepts': top_concepts.tolist(),
            'weights': concept_weights.tolist()
        })
        
        # Log region activations
        if region_features:
            region_acts = {}
            for region, features in region_features.items():
                activation = features[0].abs().mean().item()
                region_acts[region] = activation
            
            self.region_logs.append({
                'step': self.global_step,
                'activations': region_acts
            })
    
    def save_interpretability_logs(self, output_dir):
        """Save interpretability logs"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if self.concept_logs:
            with open(output_dir / 'concept_logs.json', 'w') as f:
                json.dump(self.concept_logs, f, indent=2)
            print(f"  ✓ Saved concept logs ({len(self.concept_logs)} entries)")
        
        if self.region_logs:
            with open(output_dir / 'region_logs.json', 'w') as f:
                json.dump(self.region_logs, f, indent=2)
            print(f"  ✓ Saved region logs ({len(self.region_logs)} entries)")


def main():
    """Main training loop"""
    
    print("\n" + "="*70)
    print("TRAINING ADVANCED MAMBA-GRAPH-SEMANTIC EEG2TEXT")
    print("="*70)
    
    # Setup
    set_seed(Config.SEED)
    device = get_device_info()
    
    # Load data
    print("\n📂 Loading data...")
    all_data = load_preprocessed_data()
    train_data, val_data, test_data = split_data(
        all_data,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=Config.SEED,
        subject_wise=True
    )
    
    # Tokenizer
    print("\n🔤 Loading tokenizer...")
    tokenizer = BartTokenizer.from_pretrained(Config.BART_MODEL)
    
    # Dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_data,
        val_data,
        test_data,
        tokenizer=tokenizer,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        for_pretraining=False
    )
    
    print(f"  Train: {len(train_data)} samples")
    print(f"  Val: {len(val_data)} samples")
    print(f"  Test: {len(test_data)} samples")
    
    # Model
    print("\n🧠 Initializing model...")
    model = MambaGraphEEG2TEXT(
        channel_groups=Config.CHANNEL_GROUPS,
        hidden_dim=Config.HIDDEN_DIM,
        mamba_layers=4,
        gnn_layers=3,
        num_concepts=64,
        bart_model=Config.BART_MODEL,
        use_lora=Config.USE_LORA,
        dropout=Config.DROPOUT
    )
    
    total_params, trainable_params = count_parameters(model)
    print(f"\n📊 Model Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # Trainer
    trainer = AdvancedTrainer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        learning_rate=Config.LEARNING_RATE,
        warmup_steps=Config.WARMUP_STEPS,
        gradient_clip=Config.GRADIENT_CLIP,
        semantic_loss_weight=0.0,  # Set to 0.1-0.3 if using semantic loss
        log_interpretability=True
    )
    
    # Training loop
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)
    
    save_dir = Config.MODEL_SAVE_DIR / 'advanced_training'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    
    for epoch in range(1, Config.TRAIN_EPOCHS + 1):
        print(f"\n📅 Epoch {epoch}/{Config.TRAIN_EPOCHS}")
        
        # Train
        train_metrics = trainer.train_epoch(
            train_loader,
            epoch,
            accumulation_steps=Config.GRADIENT_ACCUMULATION_STEPS
        )
        
        if train_metrics is None:
            print("❌ Training failed due to NaN")
            break
        
        # Validate
        val_loss = trainer.validate(val_loader)
        
        # Print summary
        print(f"\n📊 Epoch {epoch} Summary:")
        print(f"  Train Loss: {train_metrics['loss']:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        if train_metrics['semantic_loss'] > 0:
            print(f"  Semantic Loss: {train_metrics['semantic_loss']:.4f}")
        
        # Save best model
        if val_loss < trainer.best_val_loss:
            trainer.best_val_loss = val_loss
            checkpoint_path = save_dir / 'best_model.pt'
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_metrics['loss'],
                'best_val_loss': trainer.best_val_loss
            }, checkpoint_path)
            
            print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
        
        # Save checkpoint every 5 epochs
        if epoch % 5 == 0:
            checkpoint_path = save_dir / f'checkpoint_epoch_{epoch}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_metrics['loss']
            }, checkpoint_path)
            print(f"  ✓ Saved checkpoint")
    
    # Save interpretability logs
    print("\n💾 Saving interpretability logs...")
    trainer.save_interpretability_logs(save_dir / 'interpretability')
    
    # Training summary
    total_time = time.time() - start_time
    print(f"\n" + "="*70)
    print("TRAINING COMPLETE!")
    print(f"Total time: {format_time(total_time)}")
    print(f"Best val loss: {trainer.best_val_loss:.4f}")
    print(f"Model saved to: {save_dir}")
    print("="*70)


if __name__ == '__main__':
    main()
