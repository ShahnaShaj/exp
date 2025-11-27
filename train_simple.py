"""
Training script for Simple EEG2Text Model

Two-stage training (following current research):
1. Pre-train encoder on masked EEG reconstruction
2. Fine-tune end-to-end on EEG-text pairs
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU (remove for GPU)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BartTokenizer
from tqdm import tqdm
import time
from pathlib import Path

from config import Config
from models_simple import SimpleEEG2Text, count_parameters
from dataset import load_preprocessed_data, split_data, create_dataloaders
from utils import (
    set_seed, AverageMeter, save_checkpoint,
    get_device_info, WarmupLinearSchedule, format_time
)


class SimpleTrainer:
    """Trainer for simple EEG2Text model"""
    
    def __init__(
        self,
        model,
        tokenizer,
        device,
        learning_rate=1e-5,
        warmup_steps=500,
        weight_decay=0.01,
        gradient_clip=0.5
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
        
        # Scheduler
        self.scheduler = WarmupLinearSchedule(
            self.optimizer,
            warmup_steps=warmup_steps,
            total_steps=100000
        )
        
        self.global_step = 0
        self.best_val_loss = float('inf')
    
    def pretrain_epoch(self, train_loader, epoch, mask_ratio=0.15):
        """Pre-train encoder with masked reconstruction"""
        self.model.train()
        losses = AverageMeter()
        
        progress_bar = tqdm(train_loader, desc=f"Pre-train Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            eeg = batch['eeg'].to(self.device)
            
            # Masked reconstruction
            loss = self.model.pretrain_encoder(eeg, mask_ratio=mask_ratio)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.gradient_clip
            )
            self.optimizer.step()
            self.scheduler.step()
            
            losses.update(loss.item())
            self.global_step += 1
            
            progress_bar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })
            
            if torch.isnan(loss):
                print(f"\n❌ NaN detected!")
                return None
        
        return losses.avg
    
    def train_epoch(self, train_loader, epoch, accumulation_steps=8):
        """Fine-tune end-to-end on EEG-text pairs"""
        self.model.train()
        losses = AverageMeter()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            eeg = batch['eeg'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward
            outputs = self.model(
                eeg=eeg,
                decoder_input_ids=input_ids,
                decoder_attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss / accumulation_steps
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
            
            losses.update(loss.item() * accumulation_steps)
            
            progress_bar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })
            
            if torch.isnan(loss):
                print(f"\n❌ NaN detected!")
                return None
        
        return losses.avg
    
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
                
                losses.update(outputs.loss.item())
        
        return losses.avg


def main():
    """Main training loop"""
    
    print("\n" + "="*70)
    print("TRAINING SIMPLE EEG2TEXT MODEL")
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
    model = SimpleEEG2Text(
        num_eeg_channels=105,
        encoder_dim=256,
        num_encoder_layers=6,
        llm_model=Config.BART_MODEL,
        use_lora=Config.USE_LORA,
        freeze_llm=False,
        dropout=Config.DROPOUT
    )
    
    total_params, trainable_params = count_parameters(model)
    print(f"\n📊 Model Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # Trainer
    trainer = SimpleTrainer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        learning_rate=Config.LEARNING_RATE,
        warmup_steps=Config.WARMUP_STEPS,
        gradient_clip=Config.GRADIENT_CLIP
    )
    
    save_dir = Config.MODEL_SAVE_DIR / 'simple_training'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # ======================================================================
    # STAGE 1: Pre-train encoder (Optional but recommended)
    # ======================================================================
    
    pretrain_epochs = 5  # Set to 0 to skip pre-training
    
    if pretrain_epochs > 0:
        print("\n" + "="*70)
        print(f"STAGE 1: PRE-TRAINING ENCODER ({pretrain_epochs} epochs)")
        print("="*70)
        
        for epoch in range(1, pretrain_epochs + 1):
            print(f"\n📅 Pre-train Epoch {epoch}/{pretrain_epochs}")
            
            pretrain_loss = trainer.pretrain_epoch(train_loader, epoch)
            
            if pretrain_loss is None:
                print("❌ Pre-training failed")
                break
            
            print(f"  Pre-train Loss: {pretrain_loss:.4f}")
            
            # Save pre-trained encoder
            if epoch == pretrain_epochs:
                checkpoint_path = save_dir / 'pretrained_encoder.pt'
                torch.save({
                    'epoch': epoch,
                    'encoder_state_dict': model.eeg_encoder.state_dict(),
                    'loss': pretrain_loss
                }, checkpoint_path)
                print(f"  ✓ Saved pre-trained encoder")
    
    # ======================================================================
    # STAGE 2: Fine-tune end-to-end
    # ======================================================================
    
    print("\n" + "="*70)
    print(f"STAGE 2: FINE-TUNING END-TO-END ({Config.TRAIN_EPOCHS} epochs)")
    print("="*70)
    
    start_time = time.time()
    
    for epoch in range(1, Config.TRAIN_EPOCHS + 1):
        print(f"\n📅 Epoch {epoch}/{Config.TRAIN_EPOCHS}")
        
        # Train
        train_loss = trainer.train_epoch(
            train_loader,
            epoch,
            accumulation_steps=Config.GRADIENT_ACCUMULATION_STEPS
        )
        
        if train_loss is None:
            print("❌ Training failed")
            break
        
        # Validate
        val_loss = trainer.validate(val_loader)
        
        print(f"\n📊 Epoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < trainer.best_val_loss:
            trainer.best_val_loss = val_loss
            checkpoint_path = save_dir / 'best_model.pt'
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
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
                'train_loss': train_loss
            }, checkpoint_path)
            print(f"  ✓ Saved checkpoint")
    
    # Summary
    total_time = time.time() - start_time
    print(f"\n" + "="*70)
    print("TRAINING COMPLETE!")
    print(f"Total time: {format_time(total_time)}")
    print(f"Best val loss: {trainer.best_val_loss:.4f}")
    print(f"Model saved to: {save_dir}")
    print("="*70)


if __name__ == '__main__':
    main()
