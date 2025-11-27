"""
"""Pre-training script for EEG2TEXT
Self-supervised pre-training with masked EEG reconstruction
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU only

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time
from pathlib import Path

from config import Config
from models import EEGPreTraining
from dataset import load_preprocessed_data, split_data, create_dataloaders
from utils import (
    set_seed, AverageMeter, save_checkpoint, get_device_info,
    WarmupLinearSchedule, EarlyStopping, format_time, count_parameters
)


class PreTrainer:
    """Pre-training manager for EEG2TEXT"""
    
    def __init__(
        self,
        model,
        device,
        learning_rate=5e-5,
        warmup_steps=500,
        weight_decay=0.01,
        gradient_clip=1.0,
        mask_ratio=0.15,
        mask_strategy='remask'
    ):
        self.model = model.to(device)
        self.device = device
        self.gradient_clip = gradient_clip
        self.mask_ratio = mask_ratio
        self.mask_strategy = mask_strategy
        
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
        """Train for one epoch"""
        self.model.train()
        
        loss_meter = AverageMeter()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", mininterval=1.0)  # Update less frequently
        
        for batch_idx, eeg in enumerate(pbar):
            eeg = eeg.to(self.device, non_blocking=True)
            
            # Forward pass
            with torch.set_grad_enabled(True):
                loss, reconstructed = self.model(
                    eeg,
                    mask_ratio=self.mask_ratio,
                    mask_strategy=self.mask_strategy
                )
            
            # Backward pass
            self.optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip
                )
            
            self.optimizer.step()
            
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Update metrics
            loss_meter.update(loss.item(), eeg.size(0))
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss_meter.avg:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
        
        return loss_meter.avg
    
    def validate(self, val_loader):
        """Validate the model - memory efficient version"""
        self.model.eval()
        
        loss_meter = AverageMeter()
        
        with torch.no_grad(), torch.inference_mode():  # Faster inference
            for eeg in tqdm(val_loader, desc="Validating", mininterval=1.0):
                try:
                    eeg = eeg.to(self.device, non_blocking=True)
                    
                    # Process in smaller chunks if batch is large
                    if eeg.size(0) > 8:
                        losses = []
                        for i in range(0, eeg.size(0), 4):
                            chunk = eeg[i:i+4]
                            loss, _ = self.model(
                                chunk,
                                mask_ratio=self.mask_ratio,
                                mask_strategy=self.mask_strategy
                            )
                            losses.append(loss.item() * chunk.size(0))
                        loss_val = sum(losses) / eeg.size(0)
                        loss_meter.update(loss_val, eeg.size(0))
                    else:
                        loss, _ = self.model(
                            eeg,
                            mask_ratio=self.mask_ratio,
                            mask_strategy=self.mask_strategy
                        )
                        loss_meter.update(loss.item(), eeg.size(0))
                    
                    # Clear cache periodically
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                except RuntimeError as e:
                    if "out of memory" in str(e) or "not enough memory" in str(e):
                        print(f"\n⚠ Memory error, skipping batch")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
        
        return loss_meter.avg
    
    def train(
        self,
        train_loader,
        val_loader,
        num_epochs,
        save_dir,
        log_dir=None,
        early_stopping_patience=10,
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
        
        best_val_loss = best_val_loss_init if best_val_loss_init is not None else float('inf')
        
        print("\n" + "="*70)
        print("STARTING PRE-TRAINING" if start_epoch == 1 else "RESUMING PRE-TRAINING")
        print("="*70)
        print(f"Epochs: {start_epoch} to {num_epochs}")
        print(f"Training batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")
        print(f"Mask strategy: {self.mask_strategy}")
        print(f"Mask ratio: {self.mask_ratio}")
        if start_epoch > 1:
            print(f"Best val loss so far: {best_val_loss:.4f}")
        print("="*70)
        
        start_time = time.time()
        
        for epoch in range(start_epoch, num_epochs + 1):
            epoch_start_time = time.time()
            
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss = self.validate(val_loader)
            
            epoch_time = time.time() - epoch_start_time
            
            # Print epoch results
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Time: {format_time(epoch_time)}")
            
            # Log to TensorBoard
            if writer:
                writer.add_scalar('pretrain/train_loss', train_loss, epoch)
                writer.add_scalar('pretrain/val_loss', val_loss, epoch)
                writer.add_scalar('pretrain/lr', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'best_val_loss': best_val_loss,
                    'config': {
                        'mask_ratio': self.mask_ratio,
                        'mask_strategy': self.mask_strategy,
                        'learning_rate': self.optimizer.param_groups[0]['lr']
                    }
                }
                save_checkpoint(checkpoint, save_dir / 'best_pretrain.pt')
                print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
            
            # Save checkpoint after EVERY epoch (for crash recovery)
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'config': {
                    'mask_ratio': self.mask_ratio,
                    'mask_strategy': self.mask_strategy,
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                }
            }
            save_checkpoint(checkpoint, save_dir / 'last_pretrain_checkpoint.pt')
            
            # Save periodic numbered checkpoint
            if epoch % Config.SAVE_EVERY_N_EPOCHS == 0:
                save_checkpoint(checkpoint, save_dir / f'pretrain_epoch_{epoch}.pt')
                print(f"  ✓ Saved checkpoint for epoch {epoch}")
            
            # Early stopping
            if early_stopping(val_loss):
                print(f"\n⚠ Early stopping triggered after {epoch} epochs")
                break
        
        total_time = time.time() - start_time
        
        print("\n" + "="*70)
        print("PRE-TRAINING COMPLETED")
        print("="*70)
        print(f"Total time: {format_time(total_time)}")
        print(f"Best val loss: {best_val_loss:.4f}")
        print(f"Model saved to: {save_dir}")
        print("="*70)
        
        if writer:
            writer.close()
        
        return best_val_loss


def main():
    """Main pre-training function"""
    
    # CPU optimizations
    if Config.ENABLE_CPU_OPTIMIZATIONS and not torch.cuda.is_available():
        print("🔧 Applying CPU optimizations...")
        torch.set_num_threads(4)  # Limit CPU threads for better performance
        torch.set_num_interop_threads(2)  # Limit inter-op parallelism
        torch.backends.mkldnn.enabled = True  # Intel MKL-DNN optimization
        print("  ✓ CPU threading optimized")
        print("  ✓ MKL-DNN enabled")
    
    # Set seed for reproducibility
    set_seed(Config.SEED)
    
    # Get device
    device = get_device_info()
    
    # Print configuration
    Config.print_config()
    
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
        batch_size=Config.PRETRAIN_BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        for_pretraining=True
    )
    
    print(f"\n✓ Data loaded successfully")
    
    # Initialize model
    print("\n" + "="*70)
    print("INITIALIZING MODEL")
    print("="*70)
    
    model = EEGPreTraining(
        num_channels=Config.NUM_CHANNELS,
        hidden_dim=Config.HIDDEN_DIM,
        num_layers=Config.NUM_TRANSFORMER_LAYERS,
        num_heads=Config.NUM_ATTENTION_HEADS,
        ff_dim=Config.FEEDFORWARD_DIM,
        dropout=Config.DROPOUT
    )
    
    print(f"✓ Model initialized")
    
    # Initialize trainer
    trainer = PreTrainer(
        model=model,
        device=device,
        learning_rate=Config.PRETRAIN_LR,
        warmup_steps=Config.WARMUP_STEPS,
        weight_decay=Config.WEIGHT_DECAY,
        gradient_clip=Config.GRADIENT_CLIP,
        mask_ratio=Config.MASK_RATIO,
        mask_strategy=Config.MASK_STRATEGY
    )
    
    # Check for existing checkpoint to resume from
    save_dir = Config.MODEL_SAVE_DIR / 'pretraining'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Try to resume from last checkpoint first (most recent)
    last_checkpoint_path = save_dir / 'last_pretrain_checkpoint.pt'
    best_checkpoint_path = save_dir / 'best_pretrain.pt'
    
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
            
            # Note: scheduler state not saved (doesn't support state_dict)
            
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
        num_epochs=Config.PRETRAIN_EPOCHS,
        save_dir=save_dir,
        log_dir=Config.OUTPUT_DIR / 'pretrain_logs',
        early_stopping_patience=Config.EARLY_STOPPING_PATIENCE,
        start_epoch=start_epoch,
        best_val_loss_init=best_val_loss_so_far
    )
    
    print(f"\n✅ Pre-training completed! Best validation loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    main()
