"""
Configuration file for EEG2TEXT model training
Hyperparameters based on the paper: "Open Vocabulary EEG-to-Text Decoding with EEG Pre-Training"

CPU OPTIMIZATIONS APPLIED:
- Reduced model size: 256 hidden dim (from 512), 4 layers (from 6)
- Reduced sequence length: 1000 (from 2000)
- Larger batch sizes: 16 for pretrain (from 8), 8 for training (from 4)
- Fewer epochs: 5 for pretrain (from 10), 30 for training (from 50)
- Expected speedup: ~4-6x faster per epoch
- Model params reduced: ~5M (from ~21M)
"""

import torch
import os
from pathlib import Path

class Config:
    """Global configuration for EEG2TEXT"""
    
    # Auto-detect environment (Kaggle vs Local)
    IS_KAGGLE = os.path.exists('/kaggle/input')
    
    # Data paths - Automatically adapts to Kaggle or Local
    #if IS_KAGGLE:
    print("🔍 Detected Kaggle environment - using Kaggle paths")
    ZUCO_ROOT = Path('/kaggle/input/zuco-raw')
    PROCESSED_DATA_DIR = Path('/kaggle/input/zucopickle')
    OUTPUT_DIR = Path('/kaggle/working/outputs')
    MODEL_SAVE_DIR = Path('/kaggle/working/models')
    RESULTS_DIR = Path('/kaggle/working/results')
   ''' else:
        # Local paths (Windows/Mac/Linux)
        ZUCO_ROOT = Path('./zuco_data')  # Contains task1-SR, task2-NR, task3-TSR
        PROCESSED_DATA_DIR = Path('./processed_zuco')
        OUTPUT_DIR = Path('./outputs')
        MODEL_SAVE_DIR = Path('./models')
        RESULTS_DIR = Path('./results')'''
    
    # Create directories
    for dir_path in [PROCESSED_DATA_DIR, OUTPUT_DIR, MODEL_SAVE_DIR, RESULTS_DIR]:
        dir_path.mkdir(exist_ok=True, parents=True)
    
    # Dataset configuration
    NUM_CHANNELS = 105  # ZuCo EEG channels
    SAMPLING_RATE = 500  # Hz
    MAX_EEG_LENGTH = 800  # Further reduced for CPU (was 2000, then 1000)
    MAX_TEXT_LENGTH = 128  # Maximum text sequence length
    
    # Model architecture (optimized for CPU)
    HIDDEN_DIM = 128  # Further reduced from 256
    NUM_TRANSFORMER_LAYERS = 2  # Further reduced from 4
    NUM_ATTENTION_HEADS = 4  # Keep at 4
    FEEDFORWARD_DIM = 512  # Further reduced from 1024
    DROPOUT = 0.1
    
    # Use smaller BART for CPU
    USE_SMALL_BART = True  # Set to False if you have enough RAM
    
    # Pre-training configuration (optimized for CPU)
    PRETRAIN_EPOCHS = 10  # Paper recommendation (use 5 if extremely time-constrained)
    PRETRAIN_BATCH_SIZE = 4  # Further reduced for CPU memory (was 8)
    PRETRAIN_LR = 5e-5
    MASK_RATIO = 0.15
    MASK_STRATEGY = 'remask'  # Options: 'remask', 'masked', 'continuous'
    
    # Main training configuration (optimized for CPU)
    TRAIN_EPOCHS = 30  # Reduced from 50
    BATCH_SIZE = 1  # Further reduced for CPU memory (was 2)
    LEARNING_RATE = 1e-5  # Reduced from 3e-5 to prevent NaN
    WARMUP_STEPS = 500  # Increased warmup for stability
    WEIGHT_DECAY = 0.01
    GRADIENT_CLIP = 0.5  # Reduced from 1.0 for more aggressive clipping
    
    # BART configuration
    BART_MODEL = 'sshleifer/distilbart-cnn-6-6'  # 50% smaller than bart-base, for CPU
    # Alternative: 'facebook/bart-base' if you have more RAM
    
    # LoRA configuration (reduces memory by 98%)
    USE_LORA = True  # Enable LoRA for BART fine-tuning
    LORA_R = 8  # Rank of LoRA matrices (4, 8, or 16)
    LORA_ALPHA = 32  # Scaling factor (typically 2x or 4x of r)
    LORA_DROPOUT = 0.1
    LORA_TARGET_MODULES = ["q_proj", "v_proj"]  # Which BART layers to adapt
    
    # Training settings
    DEVICE = torch.device('cpu')
    NUM_WORKERS = 0  # Set to 0 for CPU/Windows to avoid multiprocessing overhead
    SEED = 42
    GRADIENT_ACCUMULATION_STEPS = 8  # Simulate batch size of 8 with actual batch size of 1
    USE_AMP = False  # Automatic Mixed Precision (only useful with CUDA)
    
    # CPU-specific optimizations
    ENABLE_CPU_OPTIMIZATIONS = True
    PIN_MEMORY = False  # Only useful with CUDA
    
    # Validation and checkpointing
    VAL_SPLIT = 0.2
    TEST_SPLIT = 0.1
    SAVE_EVERY_N_EPOCHS = 5
    EARLY_STOPPING_PATIENCE = 10
    
    # Generation settings
    MAX_GENERATION_LENGTH = 50
    NUM_BEAMS = 5
    
    # Logging
    LOG_INTERVAL = 50  # Log every N batches
    WANDB_PROJECT = 'eeg2text'  # Optional: for wandb logging
    USE_WANDB = False
    
    # ZuCo channel groups (10 brain regions) - indices 0-104
    # Note: Original paper indices adjusted to be within valid range [0, 104]
    CHANNEL_GROUPS = {
        'prefrontal': [4, 10, 3, 9, 14, 13, 18, 19, 22, 23, 24, 0, 25, 
                      26, 1, 2, 21, 17, 20, 7, 8, 16, 31, 15, 6],
        'premotor': [5, 104, 103, 102, 101, 100, 99, 98, 97, 96, 
                    95, 94, 93, 11, 28],
        'broca': [27, 34, 33, 32],
        'auditory_assoc': [38, 36, 37, 41, 42, 44, 55, 56, 62],
        'primary_motor': [29, 78, 53, 35, 85, 91, 92, 90, 89],
        'primary_sensory': [52, 77, 59, 76, 60, 51, 84, 83, 82, 81, 80],
        'somatic_sensory': [65, 75, 69, 70, 74, 64, 79, 58, 71],
        'auditory': [57, 88, 87, 49],
        'wernicke': [39, 40, 50, 45, 43, 48],
        'visual': [63, 67, 68, 72, 73, 86, 54, 61, 66, 46, 47]
    }
    
    # ZuCo tasks
    TASKS = ['task1-SR', 'task2-NR', 'task3-TSR']
    SUBJECT_CODES = ['ZAB', 'ZDM', 'ZDN', 'ZGW', 'ZJM', 'ZJN', 
                     'ZJS', 'ZKB', 'ZKH', 'ZKW', 'ZMG', 'ZPH']
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("=" * 60)
        print("EEG2TEXT Configuration")
        print("=" * 60)
        print(f"Device: {cls.DEVICE}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Learning Rate: {cls.LEARNING_RATE}")
        print(f"Hidden Dim: {cls.HIDDEN_DIM}")
        print(f"Num Transformer Layers: {cls.NUM_TRANSFORMER_LAYERS}")
        print(f"Pre-training Epochs: {cls.PRETRAIN_EPOCHS}")
        print(f"Main Training Epochs: {cls.TRAIN_EPOCHS}")
        print(f"BART Model: {cls.BART_MODEL}")
        print("=" * 60)
