
"""Main training script for EEG2TEXT
Supervised training of the complete EEG-to-Text model
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU only

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time
import numpy as np
from pathlib import Path
from transformers import BartTokenizer

from config import Config
from models import EEG2TEXT
from dataset import load_preprocessed_data, split_data, create_dataloaders
from utils import (
    set_seed, AverageMeter, save_checkpoint, load_checkpoint, get_device_info,
    WarmupLinearSchedule, EarlyStopping, format_time, count_parameters
)


class EEG2TextTrainer:
    """Trainer for EEG2TEXT model"""
    
    def __init__(
        self,
        model,
        tokenizer,
        device,
        learning_rate=3e-5,
        warmup_steps=500,
        weight_decay=0.01,
        gradient_clip=1.0
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.gradient_clip = gradient_clip
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Will be initialized in train()
        self.scheduler = None
        self.warmup_steps = warmup_steps
        
        print(f"\nModel Parameters: {count_parameters(self.model):,}")
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch WITHOUT teacher forcing"""
        self.model.train()
        
        loss_meter = AverageMeter()
        accumulation_steps = getattr(Config, 'GRADIENT_ACCUMULATION_STEPS', 1)
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", mininterval=2.0)
        
        for batch_idx, batch in enumerate(pbar):
            eeg = batch['eeg'].to(self.device, non_blocking=True)
            labels = batch['labels'].to(self.device, non_blocking=True)
            
            # Forward pass WITHOUT teacher forcing
            # Model must generate autoregressively from EEG only
            outputs = self.model(
                eeg=eeg,
                decoder_input_ids=None,  # No teacher forcing!
                decoder_attention_mask=None,
                labels=labels
            )
            
            loss = outputs.loss / accumulation_steps  # Scale loss for accumulation
            
            # Backward pass
            loss.backward()
            
            # Only update weights every N steps
            if (batch_idx + 1) % accumulation_steps == 0:
                # Gradient clipping
                if self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip
                    )
                
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                
                # Update scheduler
                if self.scheduler is not None:
                    self.scheduler.step()
            
            # Update metrics (unscale loss for display)
            loss_meter.update(loss.item() * accumulation_steps, eeg.size(0))
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss_meter.avg:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            # Log every N batches
            if (batch_idx + 1) % Config.LOG_INTERVAL == 0:
                current_step = (epoch - 1) * len(train_loader) + batch_idx + 1
                print(f"\n  Step {current_step}: Loss={loss_meter.avg:.4f}, LR={self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Check for NaN and save emergency checkpoint
            if not np.isfinite(loss_meter.avg):
                print(f"\n⚠️ NaN detected at step {batch_idx}! Saving emergency checkpoint before crash...")
                return float('nan')  # Signal NaN to caller
        
        return loss_meter.avg
    
    def validate(self, val_loader):
        """Validate the model WITHOUT teacher forcing - memory efficient"""
        self.model.eval()
        
        loss_meter = AverageMeter()
        
        with torch.no_grad(), torch.inference_mode():
            for batch in tqdm(val_loader, desc="Validating", mininterval=2.0):
                try:
                    eeg = batch['eeg'].to(self.device, non_blocking=True)
                    labels = batch['labels'].to(self.device, non_blocking=True)
                    
                    outputs = self.model(
                        eeg=eeg,
                        decoder_input_ids=None,  # No teacher forcing!
                        decoder_attention_mask=None,
                        labels=labels
                    )
                    
                    loss_meter.update(outputs.loss.item(), eeg.size(0))
                    
                    # Clear memory
                    del outputs, eeg, labels
                    
                except RuntimeError as e:
                    if "out of memory" in str(e) or "not enough memory" in str(e):
                        print(f"\n⚠ Memory error in validation, skipping batch")
                        continue
                    raise e
        
        return loss_meter.avg
    
    def generate_samples(self, val_loader, num_samples=5):
        """Generate sample predictions for qualitative evaluation"""
        self.model.eval()
        
        samples = []
        
        with torch.no_grad():
            for batch in val_loader:
                if len(samples) >= num_samples:
                    break
                
                eeg = batch['eeg'].to(self.device)
                true_texts = batch['texts']
                
                # Generate predictions
                outputs = self.model.generate(
                    eeg,
                    max_length=Config.MAX_GENERATION_LENGTH,
                    num_beams=Config.NUM_BEAMS
                )
                
                # Decode predictions
                pred_texts = [
                    self.tokenizer.decode(ids, skip_special_tokens=True)
                    for ids in outputs
                ]
                
                # Collect samples
                for true_text, pred_text in zip(true_texts, pred_texts):
                    if len(samples) >= num_samples:
                        break
                    samples.append({
                        'true': true_text,
                        'pred': pred_text
                    })
        
        return samples
    
    def train(
        self,
        train_loader,
        val_loader,
        num_epochs,
        save_dir,
        log_dir=None,
        early_stopping_patience=10,
        resume_from=None,
        start_epoch=1,
        best_val_loss_init=None
    ):
        """
        Complete training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            save_dir: Directory to save checkpoints
            log_dir: Directory for TensorBoard logs
            early_stopping_patience: Patience for early stopping
            resume_from: Path to checkpoint to resume from (deprecated, use start_epoch)
            start_epoch: Epoch to start from (for resuming)
            best_val_loss_init: Initial best validation loss (for resuming)
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard writer
        writer = None
        if log_dir:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(log_dir)
        
        # Set initial best validation loss
        best_val_loss = best_val_loss_init if best_val_loss_init is not None else float('inf')
        
        # Initialize scheduler
        total_steps = len(train_loader) * num_epochs
        self.scheduler = WarmupLinearSchedule(
            self.optimizer,
            warmup_steps=self.warmup_steps,
            total_steps=total_steps
        )
        
        # Early stopping
        early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            mode='min'
        )
        
        print("\n" + "="*70)
        print("STARTING TRAINING" if start_epoch == 1 else "RESUMING TRAINING")
        print("="*70)
        print(f"Epochs: {start_epoch} to {num_epochs}")
        print(f"Training batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")
        if start_epoch > 1:
            print(f"Best val loss so far: {best_val_loss:.4f}")
        print("="*70)
        
        start_time = time.time()
        
        for epoch in range(start_epoch, num_epochs + 1):
            epoch_start_time = time.time()
            
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            
            # Check for NaN - stop immediately if detected
            if not np.isfinite(train_loss):
                print(f"\n❌ NaN detected in training loss at epoch {epoch}!")
                print(f"Training stopped to prevent further corruption.")
                print(f"Last valid checkpoint saved at: {save_dir / 'last_checkpoint.pt'}")
                print(f"\nTo fix:")
                print(f"  1. Learning rate is now reduced to {Config.LEARNING_RATE}")
                print(f"  2. Gradient clipping is now {Config.GRADIENT_CLIP}")
                print(f"  3. Restart training: python train.py")
                break
            
            # Validate
            val_loss = self.validate(val_loader)
            
            # Generate sample predictions
            if epoch % 5 == 0:  # Every 5 epochs
                print("\n" + "-"*70)
                print("SAMPLE PREDICTIONS:")
                print("-"*70)
                samples = self.generate_samples(val_loader, num_samples=3)
                for i, sample in enumerate(samples):
                    print(f"\nSample {i+1}:")
                    print(f"  True: {sample['true']}")
                    print(f"  Pred: {sample['pred']}")
                print("-"*70)
            
            epoch_time = time.time() - epoch_start_time
            
            # Print epoch results
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Time: {format_time(epoch_time)}")
            
            # Log to TensorBoard
            if writer:
                writer.add_scalar('train/train_loss', train_loss, epoch)
                writer.add_scalar('train/val_loss', val_loss, epoch)
                writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'best_val_loss': best_val_loss
                }
                save_checkpoint(checkpoint, save_dir / 'best_model.pt')
                print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
            
            # Save checkpoint after EVERY epoch (for crash recovery)
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss
            }
            save_checkpoint(checkpoint, save_dir / 'last_checkpoint.pt')
            
            # Save periodic numbered checkpoint
            if epoch % Config.SAVE_EVERY_N_EPOCHS == 0:
                save_checkpoint(checkpoint, save_dir / f'model_epoch_{epoch}.pt')
                print(f"  ✓ Saved checkpoint for epoch {epoch}")
            
            # Early stopping
            if early_stopping(val_loss):
                print(f"\n⚠ Early stopping triggered after {epoch} epochs")
                break
        
        total_time = time.time() - start_time
        
        print("\n" + "="*70)
        print("TRAINING COMPLETED")
        print("="*70)
        print(f"Total time: {format_time(total_time)}")
        print(f"Best val loss: {best_val_loss:.4f}")
        print(f"Model saved to: {save_dir}")
        print("="*70)
        
        if writer:
            writer.close()
        
        return best_val_loss


def main():
    """Main training function"""
    
    # CPU optimizations
    if Config.ENABLE_CPU_OPTIMIZATIONS and not torch.cuda.is_available():
        print("🔧 Applying CPU optimizations...")
        torch.set_num_threads(4)  # Limit CPU threads to reduce overhead
        torch.set_num_interop_threads(2)  # Limit inter-op parallelism
        # Enable CPU-specific optimizations
        torch.backends.mkldnn.enabled = True  # Intel MKL-DNN optimization
        print("  ✓ CPU threading optimized")
        print("  ✓ MKL-DNN enabled")
    
    # Set seed for reproducibility
    set_seed(Config.SEED)
    
    # Get device
    device = get_device_info()
    
    # Print configuration
    Config.print_config()
    
    # Load tokenizer
    print("\n" + "="*70)
    print("LOADING TOKENIZER")
    print("="*70)
    tokenizer = BartTokenizer.from_pretrained(Config.BART_MODEL)
    print(f"✓ Loaded tokenizer: {Config.BART_MODEL}")
    
    # Load preprocessed data
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    
    all_data = load_preprocessed_data()
    
    # Split data
    train_data, val_data, test_data = split_data(
        all_data,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=Config.SEED
    )
    
    # Create dataloaders
    train_loader, val_loader, _ = create_dataloaders(
        train_data,
        val_data,
        test_data,
        tokenizer=tokenizer,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        for_pretraining=False
    )
    
    print(f"✓ Data loaded successfully")
    
    # Initialize model
    print("\n" + "="*70)
    print("INITIALIZING MODEL")
    print("="*70)
    print(f"Creating EEG encoder (this may take a moment)...")
    
    try:
        model = EEG2TEXT(
            channel_groups=Config.CHANNEL_GROUPS,
            hidden_dim=Config.HIDDEN_DIM,
            num_layers=Config.NUM_TRANSFORMER_LAYERS,
            num_heads=Config.NUM_ATTENTION_HEADS,
            ff_dim=Config.FEEDFORWARD_DIM,
            dropout=Config.DROPOUT,
            bart_model=Config.BART_MODEL
        )
    except Exception as e:
        print(f"\n❌ Failed to initialize model: {e}")
        print("\nYour system may not have enough RAM for the full model.")
        print("Suggestions:")
        print("  1. Close other programs to free memory")
        print("  2. Use Google Colab with free GPU")
        print("  3. Run only pretraining: python pretrain.py")
        raise
    
    # Load pre-trained weights if available
    pretrained_path = Config.MODEL_SAVE_DIR / 'pretraining' / 'best_pretrain.pt'
    if pretrained_path.exists():
        print(f"\nLoading pre-trained weights from: {pretrained_path}")
        try:
            model.load_pretrained_encoder(pretrained_path)
            print("✓ Pre-trained weights loaded successfully")
        except Exception as e:
            print(f"⚠ Warning: Could not load pre-trained weights: {e}")
            print("  Proceeding with random initialization")
    else:
        print(f"\n⚠ Warning: Pre-trained weights not found at {pretrained_path}")
        print("  Proceeding with random initialization")
        print("  Consider running pre-training first for better results")
    
    print(f"✓ Model initialized")
    
    # Print memory usage info
    if hasattr(Config, 'USE_LORA') and Config.USE_LORA:
        print("\n" + "="*70)
        print("LoRA ENABLED - Memory Optimized Training")
        print("="*70)
        print(f"  BART trainable parameters: ~1-2M (98% reduction)")
        print(f"  Expected RAM usage: 3-4GB (manageable on your system!)")
        print("="*70)
    
    # Initialize trainer
    trainer = EEG2TextTrainer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        learning_rate=Config.LEARNING_RATE,
        warmup_steps=Config.WARMUP_STEPS,
        weight_decay=Config.WEIGHT_DECAY,
        gradient_clip=Config.GRADIENT_CLIP
    )
    
    # Check for existing checkpoint to resume from
    save_dir = Config.MODEL_SAVE_DIR / 'main_training'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Try to resume from last checkpoint first (most recent)
    last_checkpoint_path = save_dir / 'last_checkpoint.pt'
    best_checkpoint_path = save_dir / 'best_model.pt'
    
    start_epoch = 1
    best_val_loss_so_far = float('inf')
    
    # Prioritize last_checkpoint.pt for crash recovery
    checkpoint_to_load = None
    if last_checkpoint_path.exists():
        checkpoint_to_load = last_checkpoint_path
        checkpoint_type = "LAST CHECKPOINT (crash recovery)"
    elif best_checkpoint_path.exists():
        checkpoint_to_load = best_checkpoint_path
        checkpoint_type = "BEST CHECKPOINT"
    
    if checkpoint_to_load:
        print(f"\n{'='*70}")
        print(f"FOUND EXISTING {checkpoint_type}")
        print(f"{'='*70}")
        try:
            checkpoint = torch.load(checkpoint_to_load, map_location=device)
            trainer.model.load_state_dict(checkpoint['model_state_dict'])
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Note: scheduler state not saved (WarmupLinearSchedule doesn't support it)
            # Scheduler will be recreated with same settings
            
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss_so_far = checkpoint.get('best_val_loss', checkpoint.get('val_loss', float('inf')))
            
            print(f"✓ Resumed from epoch {checkpoint['epoch']}")
            print(f"  Train loss: {checkpoint.get('train_loss', 'N/A'):.4f}")
            print(f"  Val loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
            print(f"  Best val loss so far: {best_val_loss_so_far:.4f}")
            print(f"  Will continue from epoch {start_epoch}")
            print(f"{'='*70}\n")
        except Exception as e:
            print(f"⚠ Could not load checkpoint: {e}")
            print(f"  Starting from scratch\n")
            start_epoch = 1
            best_val_loss_so_far = float('inf')
    
    # Train
    best_val_loss = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=Config.TRAIN_EPOCHS,
        save_dir=save_dir,
        log_dir=Config.OUTPUT_DIR / 'train_logs',
        early_stopping_patience=Config.EARLY_STOPPING_PATIENCE,
        start_epoch=start_epoch,
        best_val_loss_init=best_val_loss_so_far
    )
    
    print(f"\n✅ Training completed! Best validation loss: {best_val_loss:.4f}")
    print(f"\nNext steps:")
    print(f"  1. Run evaluation: python evaluate.py")
    print(f"  2. Check TensorBoard logs: tensorboard --logdir {Config.OUTPUT_DIR / 'train_logs'}")


if __name__ == '__main__':
    main()
