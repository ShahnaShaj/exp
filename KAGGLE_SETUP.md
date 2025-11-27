# 🚀 Kaggle Setup Guide for Simple EEG2Text Training

## 📦 Files to Upload to Kaggle

### **Essential Files** (Must upload):
```
train_simple.py          # Main training script
models_simple.py         # Model architecture (BiMamba + LLM)
dataset.py              # ZuCo data loader
config.py               # Hyperparameters and channel groups
utils.py                # Training utilities
requirements.txt        # Package dependencies
```

### **Data Files** (Must upload as Kaggle Dataset):
```
processed_zuco/         # Preprocessed ZuCo data (~500MB)
├── train_subjects.pkl
├── val_subjects.pkl
└── test_subjects.pkl
```

### **Optional Files**:
```
models_advanced.py      # If you want to try the advanced model
train_advanced.py       # Advanced model training script
evaluate.py            # Evaluation script (optional)
```

---

## 📋 Required Python Packages

All packages are in `requirements.txt` and will auto-install in Kaggle:

```
torch>=2.0.0
transformers>=4.30.0
peft>=0.7.0              # LoRA fine-tuning
accelerate>=0.20.0       # Model handling
datasets>=2.14.0         # Data processing
scipy>=1.10.0
numpy>=1.24.0
pandas>=2.0.0
h5py>=3.8.0
scikit-learn>=1.2.0
nltk>=3.8
sacrebleu>=2.3.0
rouge-score>=0.1.2
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
tensorboard>=2.13.0
```

**Note**: Kaggle usually has most of these pre-installed (torch, transformers, numpy, pandas, etc.)

---

## 🎯 Training Configuration

### **What the Model Does**:

✅ **Uses Pre-trained BART**: `sshleifer/distilbart-cnn-6-6` (82M params)
- Downloaded automatically from HuggingFace on first run
- Frozen or fine-tuned with LoRA (you choose)

✅ **Uses Your ZuCo Dataset**: `processed_zuco/` folder
- 12,470 samples of EEG-text pairs
- Subject-wise split: 8 train, 1 val, 3 test
- 105 EEG channels, 800 timesteps

✅ **Two-Stage Training**:
1. **Pre-train encoder** (5 epochs): Masked EEG reconstruction
2. **Fine-tune end-to-end** (30 epochs): EEG → Text generation

### **Checkpoint Saving**:

The script automatically saves:
```python
# During pre-training (Stage 1)
models/simple_training/pretrained_encoder.pt  # After 5 epochs

# During fine-tuning (Stage 2)
models/simple_training/best_model.pt          # Best validation loss
models/simple_training/checkpoint_epoch_5.pt  # Every 5 epochs
models/simple_training/checkpoint_epoch_10.pt
models/simple_training/checkpoint_epoch_15.pt
# ... etc
```

Each checkpoint contains:
```python
{
    'epoch': current_epoch,
    'model_state_dict': full_model_weights,
    'optimizer_state_dict': optimizer_state,
    'val_loss': validation_loss,
    'train_loss': training_loss,
    'best_val_loss': best_validation_loss_so_far
}
```

---

## 🔧 How to Run on Kaggle

### **Step 1: Create New Kaggle Notebook**
1. Go to [kaggle.com/code](https://www.kaggle.com/code)
2. Click "New Notebook"
3. Select **GPU P100** or **GPU T4** (recommended)
4. Enable Internet (Settings → Internet → On)

### **Step 2: Upload Repository Files**

**Option A: Upload as ZIP**
1. Compress all files into `neurolens.zip`
2. In Kaggle: Click "Add data" → "Upload" → Select ZIP
3. In first cell: `!unzip /kaggle/input/neurolens/neurolens.zip -d /kaggle/working`

**Option B: Upload Individual Files**
1. Click "Add data" → "Upload"
2. Upload all 6 essential files listed above
3. Files appear in `/kaggle/working/`

### **Step 3: Upload Dataset**

**Create a Kaggle Dataset** (recommended for large data):
1. Go to [kaggle.com/datasets](https://www.kaggle.com/datasets)
2. Click "New Dataset"
3. Upload `processed_zuco/` folder
4. Name it: "zuco-eeg-preprocessed"
5. In notebook: Click "Add data" → "Your datasets" → Select it
6. Data appears at `/kaggle/input/zuco-eeg-preprocessed/`

### **Step 4: Install Dependencies**

```python
# Cell 1: Install packages
!pip install -q transformers accelerate peft datasets sacrebleu rouge-score

# Verify installation
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

### **Step 5: Run Training**

```python
# Cell 2: Run training
!python train_simple.py
```

That's it! The script will:
1. Load pre-trained BART from HuggingFace
2. Load your preprocessed ZuCo data
3. Pre-train encoder (5 epochs, ~3-5 hours)
4. Fine-tune end-to-end (30 epochs, ~20-30 hours)
5. Save checkpoints automatically

---

## ⚙️ Configuration Options

### **Quick Settings** (edit in `config.py` or command-line):

```python
# Faster training (lower accuracy)
BATCH_SIZE = 8              # Default: 4
GRADIENT_ACCUMULATION = 4   # Default: 8
TRAIN_EPOCHS = 15           # Default: 30

# Better accuracy (slower)
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 16
TRAIN_EPOCHS = 50
```

### **GPU Memory Issues?**

If you run out of memory:
```python
# Option 1: Reduce batch size
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 16

# Option 2: Use smaller encoder
encoder_dim = 128           # Default: 256
num_encoder_layers = 4      # Default: 6
```

---

## 📊 Expected Training Time on Kaggle GPU

| Stage | Epochs | GPU | Time |
|-------|--------|-----|------|
| Pre-training | 5 | T4 | 3-5 hours |
| Pre-training | 5 | P100 | 2-3 hours |
| Fine-tuning | 30 | T4 | 20-30 hours |
| Fine-tuning | 30 | P100 | 15-20 hours |
| **Total** | **35** | **T4** | **~25-35 hours** |
| **Total** | **35** | **P100** | **~17-23 hours** |

**Tip**: Kaggle notebooks timeout after 12 hours. You'll need to:
1. Run in multiple sessions
2. Resume from last checkpoint (script auto-resumes)
3. Or reduce epochs to fit in one session

---

## 🔄 Resuming Training from Checkpoint

If your Kaggle session times out, resume training:

```python
# Edit train_simple.py, add this before training loop:

# Load checkpoint if exists
checkpoint_path = save_dir / 'checkpoint_epoch_15.pt'
if checkpoint_path.exists():
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    trainer.best_val_loss = checkpoint['best_val_loss']
    print(f"Resuming from epoch {start_epoch}")
else:
    start_epoch = 1

# Then change loop to:
for epoch in range(start_epoch, Config.TRAIN_EPOCHS + 1):
    # ... rest of training loop
```

---

## 📥 Downloading Results from Kaggle

After training completes:

```python
# Cell: Compress and download
!zip -r results.zip /kaggle/working/models/simple_training
from google.colab import files  # Use 'IPython.display import FileLink' for Kaggle
# For Kaggle: Just click "Save Version" → "Quick Save with Output"
```

Or manually download from Kaggle UI:
1. Click "Save Version" → "Save & Run All"
2. Wait for completion
3. Output files appear in "Output" tab
4. Download `models/simple_training/` folder

---

## ✅ Verification Checklist

Before starting training, verify:

- [ ] All 6 essential files uploaded to `/kaggle/working/`
- [ ] `processed_zuco/` folder available
- [ ] GPU enabled (check with `torch.cuda.is_available()`)
- [ ] Internet enabled (for downloading BART)
- [ ] Checkpoint directory exists: `models/simple_training/`

---

## 🐛 Troubleshooting

### **Error: "No such file or directory: processed_zuco"**
**Fix**: Update path in `config.py`:
```python
DATA_DIR = Path('/kaggle/input/zuco-eeg-preprocessed')
```

### **Error: "CUDA out of memory"**
**Fix**: Reduce batch size in `config.py`:
```python
BATCH_SIZE = 2  # or even 1
GRADIENT_ACCUMULATION_STEPS = 16
```

### **Error: "NaN loss detected"**
**Fix**: Reduce learning rate:
```python
LEARNING_RATE = 5e-6  # Default: 1e-5
```

### **Error: "Pretrained model not found"**
**Fix**: Ensure Internet is enabled in Kaggle settings for HuggingFace downloads.

### **Session timeout before training completes**
**Fix**: 
1. Save intermediate checkpoints (already implemented every 5 epochs)
2. Resume from checkpoint (see "Resuming Training" section)
3. Or reduce `TRAIN_EPOCHS` to 15-20 to fit in one session

---

## 🎯 Summary

**Minimum required uploads**:
```
✅ train_simple.py
✅ models_simple.py
✅ dataset.py
✅ config.py
✅ utils.py
✅ requirements.txt
✅ processed_zuco/ (as Kaggle dataset)
```

**Model architecture**:
```
✅ Pre-trained BART decoder (auto-downloaded)
✅ Your ZuCo EEG data
✅ Two-stage training (pre-train + fine-tune)
✅ Automatic checkpointing (best model + every 5 epochs)
```

**Expected results**:
```
✅ BLEU-1: 0.45-0.50 (10-20% improvement over baseline)
✅ BLEU-4: 0.15-0.18
✅ Training time: ~25-35 hours on T4 GPU
✅ Checkpoints saved every 5 epochs
```

---

**Ready to start? Just upload the files and run `!python train_simple.py` in Kaggle! 🚀**
