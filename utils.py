"""
Utility functions for EEG2TEXT
"""

import torch
import numpy as np
import random
from pathlib import Path
import json


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, filepath):
    """Save model checkpoint"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, filepath)


def load_checkpoint(filepath, model, optimizer=None):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint.get('epoch', 0), checkpoint.get('best_val_loss', float('inf'))


def count_parameters(model):
    """Count number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_lr(optimizer):
    """Get current learning rate from optimizer"""
    for param_group in optimizer.param_groups:
        return param_group['lr']


class WarmupLinearSchedule:
    """Learning rate scheduler with linear warmup and linear decay"""
    
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=0):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_step = 0
    
    def step(self):
        self.current_step += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def get_lr(self):
        if self.current_step < self.warmup_steps:
            # Linear warmup
            return self.base_lr * (self.current_step / self.warmup_steps)
        else:
            # Linear decay
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return max(self.min_lr, self.base_lr * (1 - progress))


def format_time(seconds):
    """Format seconds into readable time string"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def save_metrics(metrics, filepath):
    """Save metrics to JSON file"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)


def load_metrics(filepath):
    """Load metrics from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


class EarlyStopping:
    """Early stopping handler"""
    
    def __init__(self, patience=10, min_delta=0, mode='min'):
        """
        Args:
            patience: Number of epochs with no improvement before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max'
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


def print_model_summary(model, input_shape=None):
    """Print model architecture summary"""
    print("\n" + "="*70)
    print("MODEL SUMMARY")
    print("="*70)
    
    print(f"\nTotal parameters: {count_parameters(model):,}")
    
    if input_shape:
        print(f"Input shape: {input_shape}")
    
    print("\nModel architecture:")
    print(model)
    print("="*70)


def get_device_info():
    """Get information about available devices"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        device_name = torch.cuda.get_device_name(0)
        device_count = torch.cuda.device_count()
        memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        print("\n" + "="*70)
        print("DEVICE INFORMATION")
        print("="*70)
        print(f"Device: CUDA")
        print(f"GPU: {device_name}")
        print(f"Number of GPUs: {device_count}")
        print(f"GPU Memory: {memory:.2f} GB")
        print("="*70)
    else:
        device = torch.device('cpu')
        print("\n" + "="*70)
        print("DEVICE INFORMATION")
        print("="*70)
        print("Device: CPU")
        print("⚠ Warning: CUDA not available. Training will be slow.")
        print("="*70)
    
    return device


def log_to_tensorboard(writer, metrics, step, prefix=''):
    """Log metrics to TensorBoard"""
    if writer is None:
        return
    
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            writer.add_scalar(f"{prefix}/{key}" if prefix else key, value, step)


def collate_fn_pretrain(batch):
    """Custom collate function for pre-training (handles variable length EEG)"""
    # For pre-training, batch is a list of tensors
    return torch.stack(batch)


def collate_fn_supervised(batch):
    """Custom collate function for supervised training"""
    # Batch is a list of dicts
    eeg = torch.stack([item['eeg'] for item in batch])
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    texts = [item['text'] for item in batch]
    
    return {
        'eeg': eeg,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'texts': texts
    }


# Test utilities
def test_utilities():
    """Test utility functions"""
    print("Testing utility functions...")
    
    # Test seed setting
    print("\n1. Testing seed setting...")
    set_seed(42)
    rand1 = torch.rand(5)
    set_seed(42)
    rand2 = torch.rand(5)
    assert torch.allclose(rand1, rand2), "Seed setting failed"
    print("  ✓ Seed setting works")
    
    # Test AverageMeter
    print("\n2. Testing AverageMeter...")
    meter = AverageMeter()
    meter.update(10, 1)
    meter.update(20, 1)
    assert meter.avg == 15, "AverageMeter failed"
    print(f"  ✓ AverageMeter: avg={meter.avg}")
    
    # Test time formatting
    print("\n3. Testing time formatting...")
    time_str = format_time(3665)
    print(f"  ✓ 3665 seconds = {time_str}")
    
    # Test early stopping
    print("\n4. Testing EarlyStopping...")
    early_stop = EarlyStopping(patience=3, mode='min')
    scores = [10, 9, 8.5, 8.4, 8.4, 8.4, 8.4]
    stopped = False
    for i, score in enumerate(scores):
        if early_stop(score):
            print(f"  ✓ Early stopping triggered at epoch {i+1}")
            stopped = True
            break
    assert stopped, "Early stopping failed"
    
    print("\n✓ All utility tests passed!")


if __name__ == '__main__':
    test_utilities()
