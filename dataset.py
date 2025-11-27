"""
Dataset classes for EEG2TEXT
Handles loading and batching of preprocessed EEG data
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
from pathlib import Path
from config import Config


class EEGPreTrainDataset(Dataset):
    """
    Dataset for self-supervised pre-training
    Only needs EEG data (no text labels)
    """
    
    def __init__(self, data_list, max_eeg_len=None):
        """
        Args:
            data_list: List of dicts with 'eeg' key
            max_eeg_len: Maximum EEG sequence length (for padding/truncation)
        """
        self.data = data_list
        self.max_eeg_len = max_eeg_len or Config.MAX_EEG_LENGTH
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        eeg = item['eeg']
        
        # Pad or truncate to max_len
        if len(eeg) > self.max_eeg_len:
            eeg = eeg[:self.max_eeg_len]
        else:
            pad_len = self.max_eeg_len - len(eeg)
            eeg = np.vstack([eeg, np.zeros((pad_len, eeg.shape[1]))])
        
        return torch.FloatTensor(eeg)


class EEG2TextDataset(Dataset):
    """
    Dataset for supervised EEG-to-Text training
    Includes both EEG and corresponding text
    """
    
    def __init__(self, data_list, tokenizer, max_eeg_len=None, max_text_len=None):
        """
        Args:
            data_list: List of dicts with 'eeg' and 'text' keys
            tokenizer: Hugging Face tokenizer (e.g., BartTokenizer)
            max_eeg_len: Maximum EEG sequence length
            max_text_len: Maximum text sequence length
        """
        self.data = data_list
        self.tokenizer = tokenizer
        self.max_eeg_len = max_eeg_len or Config.MAX_EEG_LENGTH
        self.max_text_len = max_text_len or Config.MAX_TEXT_LENGTH
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        eeg = item['eeg']
        text = item['text']
        
        # Process EEG: pad or truncate
        if len(eeg) > self.max_eeg_len:
            eeg = eeg[:self.max_eeg_len]
        else:
            pad_len = self.max_eeg_len - len(eeg)
            eeg = np.vstack([eeg, np.zeros((pad_len, eeg.shape[1]))])
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            max_length=self.max_text_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # For BART, we need to prepare decoder inputs
        # Decoder input is the text shifted right
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # Labels are the same as input_ids, but with padding tokens set to -100
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'eeg': torch.FloatTensor(eeg),
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'text': text  # Keep original text for evaluation
        }


def load_preprocessed_data(data_path=None, task=None):
    """
    Load preprocessed data from pickle files
    
    Args:
        data_path: Path to processed data directory
        task: Specific task to load ('task1-SR', 'task2-NR', 'task3-TSR', or None for all)
    
    Returns:
        List of data samples
    """
    data_path = Path(data_path or Config.PROCESSED_DATA_DIR)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Processed data directory not found: {data_path}")
    
    if task:
        # Load specific task
        task_file = data_path / f"{task}_processed.pkl"
        if not task_file.exists():
            raise FileNotFoundError(f"Task file not found: {task_file}")
        
        with open(task_file, 'rb') as f:
            data = pickle.load(f)
        
        print(f"Loaded {len(data)} samples from {task}")
    else:
        # Load all tasks separately and combine
        print("Loading data from individual task files...")
        all_data = []
        tasks = ['task1-SR', 'task2-NR', 'task3-TSR']
        
        for task_name in tasks:
            task_file = data_path / f"{task_name}_processed.pkl"
            if task_file.exists():
                print(f"  Loading {task_name}...")
                with open(task_file, 'rb') as f:
                    task_data = pickle.load(f)
                all_data.extend(task_data)
                print(f"    ✓ Loaded {len(task_data)} samples")
            else:
                print(f"    ⚠ Warning: {task_file} not found, skipping")
        
        if not all_data:
            raise FileNotFoundError(f"No task files found in {data_path}")
        
        print(f"\n✓ Total loaded: {len(all_data)} samples from {len([t for t in tasks if (data_path / f'{t}_processed.pkl').exists()])} tasks")
        data = all_data
    
    return data


def split_data(data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42, subject_wise=True):
    """
    Split data into train, validation, and test sets
    
    Args:
        data: List of data samples (each should have 'subject' field)
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        seed: Random seed
        subject_wise: If True, split by subjects (prevents data leakage)
    
    Returns:
        train_data, val_data, test_data
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    # Set seed for reproducibility
    np.random.seed(seed)
    
    if subject_wise:
        # Group data by subject
        subject_data = {}
        for sample in data:
            subject = sample.get('subject', 'unknown')
            if subject not in subject_data:
                subject_data[subject] = []
            subject_data[subject].append(sample)
        
        # Get unique subjects
        subjects = list(subject_data.keys())
        np.random.shuffle(subjects)
        
        # Split subjects
        num_subjects = len(subjects)
        train_end = int(num_subjects * train_ratio)
        val_end = train_end + int(num_subjects * val_ratio)
        
        train_subjects = subjects[:train_end]
        val_subjects = subjects[train_end:val_end]
        test_subjects = subjects[val_end:]
        
        # Collect data for each split
        train_data = []
        val_data = []
        test_data = []
        
        for subject in train_subjects:
            train_data.extend(subject_data[subject])
        for subject in val_subjects:
            val_data.extend(subject_data[subject])
        for subject in test_subjects:
            test_data.extend(subject_data[subject])
        
        print(f"\n{'='*70}")
        print("SUBJECT-WISE DATA SPLIT (No Data Leakage)")
        print(f"{'='*70}")
        print(f"Total subjects: {num_subjects}")
        print(f"  Train subjects: {len(train_subjects)} - {train_subjects}")
        print(f"  Val subjects: {len(val_subjects)} - {val_subjects}")
        print(f"  Test subjects: {len(test_subjects)} - {test_subjects}")
        print(f"\nSample counts:")
        print(f"  Training: {len(train_data)} samples ({len(train_data)/len(data)*100:.1f}%)")
        print(f"  Validation: {len(val_data)} samples ({len(val_data)/len(data)*100:.1f}%)")
        print(f"  Test: {len(test_data)} samples ({len(test_data)/len(data)*100:.1f}%)")
        print(f"{'='*70}")
    else:
        # Random split (old behavior - NOT RECOMMENDED)
        print("\n⚠ WARNING: Using random split - may cause data leakage!")
        indices = np.random.permutation(len(data))
        
        # Calculate split points
        train_end = int(len(data) * train_ratio)
        val_end = train_end + int(len(data) * val_ratio)
        
        # Split indices
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        
        # Create splits
        train_data = [data[i] for i in train_indices]
        val_data = [data[i] for i in val_indices]
        test_data = [data[i] for i in test_indices]
        
        print(f"\nData split:")
        print(f"  Training: {len(train_data)} samples ({train_ratio*100:.1f}%)")
        print(f"  Validation: {len(val_data)} samples ({val_ratio*100:.1f}%)")
        print(f"  Test: {len(test_data)} samples ({test_ratio*100:.1f}%)")
    
    return train_data, val_data, test_data


def create_dataloaders(
    train_data,
    val_data,
    test_data,
    tokenizer=None,
    batch_size=None,
    num_workers=None,
    for_pretraining=False
):
    """
    Create PyTorch DataLoaders
    
    Args:
        train_data: Training data list
        val_data: Validation data list
        test_data: Test data list
        tokenizer: Tokenizer (required if not for_pretraining)
        batch_size: Batch size
        num_workers: Number of data loading workers
        for_pretraining: If True, use EEGPreTrainDataset; else use EEG2TextDataset
    
    Returns:
        train_loader, val_loader, test_loader
    """
    batch_size = batch_size or Config.BATCH_SIZE
    num_workers = num_workers or Config.NUM_WORKERS
    
    if for_pretraining:
        # Pre-training datasets (no text needed)
        train_dataset = EEGPreTrainDataset(train_data)
        val_dataset = EEGPreTrainDataset(val_data)
        test_dataset = EEGPreTrainDataset(test_data)
    else:
        # Supervised training datasets
        if tokenizer is None:
            raise ValueError("tokenizer is required for supervised training")
        
        train_dataset = EEG2TextDataset(train_data, tokenizer)
        val_dataset = EEG2TextDataset(val_data, tokenizer)
        test_dataset = EEG2TextDataset(test_data, tokenizer)
    
    # Create dataloaders
    # Use config PIN_MEMORY setting if available, otherwise auto-detect
    pin_memory = getattr(Config, 'PIN_MEMORY', torch.cuda.is_available())
    
    # Use smaller batch size for validation/test to save memory
    val_test_batch_size = max(2, batch_size // 2) if batch_size > 2 else batch_size
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=val_test_batch_size,  # Smaller batch for memory efficiency
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=val_test_batch_size,  # Smaller batch for memory efficiency
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader


# ============================================================================
# TESTING
# ============================================================================

def test_datasets():
    """Test dataset loading and batching"""
    print("Testing dataset functionality...")
    
    # Create dummy data
    dummy_data = [
        {
            'eeg': np.random.randn(1500, 105),
            'text': f"This is test sentence {i}.",
            'task': 'task1-SR'
        }
        for i in range(100)
    ]
    
    # Test pre-training dataset
    print("\n1. Testing EEGPreTrainDataset...")
    pretrain_dataset = EEGPreTrainDataset(dummy_data, max_eeg_len=2000)
    sample = pretrain_dataset[0]
    print(f"  Sample shape: {sample.shape}")
    assert sample.shape == (2000, 105), "Incorrect shape"
    
    # Test supervised dataset
    print("\n2. Testing EEG2TextDataset...")
    from transformers import BartTokenizer
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    
    sup_dataset = EEG2TextDataset(dummy_data, tokenizer, max_eeg_len=2000, max_text_len=128)
    sample = sup_dataset[0]
    print(f"  EEG shape: {sample['eeg'].shape}")
    print(f"  Input IDs shape: {sample['input_ids'].shape}")
    print(f"  Labels shape: {sample['labels'].shape}")
    print(f"  Text: {sample['text']}")
    
    # Test data splitting
    print("\n3. Testing data splitting...")
    train, val, test = split_data(dummy_data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    
    # Test dataloader creation
    print("\n4. Testing DataLoader creation...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train, val, test,
        tokenizer=tokenizer,
        batch_size=4,
        num_workers=0,
        for_pretraining=False
    )
    
    # Test batching
    batch = next(iter(train_loader))
    print(f"  Batch EEG shape: {batch['eeg'].shape}")
    print(f"  Batch input_ids shape: {batch['input_ids'].shape}")
    
    print("\n✓ All dataset tests passed!")


if __name__ == '__main__':
    test_datasets()
