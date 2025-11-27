# 🚀 KAGGLE QUICK START - 3 Steps Only!

## ✅ Your config.py is Already Kaggle-Ready!

The code **auto-detects** Kaggle environment. No manual path changes needed!

---

## 📦 Step 1: Upload to Kaggle

### **Upload Your GitHub Repo**:
1. Your repo is already on GitHub: `ShahnaShaj/exp`
2. In Kaggle notebook, add this cell:

```python
# Clone your GitHub repo
!git clone https://github.com/ShahnaShaj/exp.git
!cd exp
!ls -la
```

### **Upload Your Dataset**:
1. Go to [kaggle.com/datasets](https://www.kaggle.com/datasets)
2. Click "New Dataset"
3. Upload your `processed_zuco/` folder
4. Name it: **`zuco-preprocessed`** (must match this name!)
5. Make it public or private

---

## 🔧 Step 2: Setup Kaggle Notebook

Create a new Kaggle notebook and run these cells:

```python
# =============================================================================
# CELL 1: Install dependencies
# =============================================================================
!pip install -q transformers>=4.30.0 accelerate>=0.20.0 peft>=0.7.0 \
             datasets>=2.14.0 sacrebleu>=2.3.0 rouge-score>=0.1.2

# =============================================================================
# CELL 2: Clone repo and copy files
# =============================================================================
import os
import shutil

# Clone your repo
!git clone https://github.com/ShahnaShaj/exp.git

# Copy Python files to working directory
files = [
    'train_simple.py', 'models_simple.py', 'dataset.py',
    'config.py', 'utils.py'
]

for file in files:
    shutil.copy2(f'exp/{file}', f'/kaggle/working/{file}')
    print(f'✓ Copied {file}')

os.chdir('/kaggle/working')
print(f'\n✓ Working directory: {os.getcwd()}')

# =============================================================================
# CELL 3: Add your dataset
# =============================================================================
# In Kaggle UI: Click "Add data" → "Your datasets" → Select "zuco-preprocessed"
# The dataset will appear at /kaggle/input/zuco-preprocessed/

# Verify it exists
from pathlib import Path
data_path = Path('/kaggle/input/zuco-preprocessed')
print(f'Data exists: {data_path.exists()}')
if data_path.exists():
    files = list(data_path.glob('*.pkl'))
    print(f'Found {len(files)} files: {[f.name for f in files]}')

# =============================================================================
# CELL 4: Verify GPU and setup
# =============================================================================
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')

# =============================================================================
# CELL 5: RUN TRAINING 🚀
# =============================================================================
!python train_simple.py

# =============================================================================
# CELL 6: Check outputs
# =============================================================================
!ls -lh /kaggle/working/models/simple_training/
!du -sh /kaggle/working/models/simple_training/
```

---

## 📊 Step 3: Monitor and Download

### **During Training**:
- Watch the progress bars in the cell output
- Training saves checkpoints every 5 epochs automatically

### **After Training**:
```python
# Check what was saved
!find /kaggle/working/models -name "*.pt" -exec ls -lh {} \;

# Compress for download
!zip -r /kaggle/working/simple_training_results.zip /kaggle/working/models/simple_training
```

### **Download Results**:
1. Click "Save Version" → "Save & Run All"
2. Wait for completion
3. Go to "Output" tab
4. Download files from `/kaggle/working/models/`

---

## 🎯 Important: Dataset Name Must Match!

Your Kaggle dataset **must** be named exactly: **`zuco-preprocessed`**

If you named it something else (e.g., `my-zuco-data`), you have 2 options:

### **Option A: Rename in Kaggle** (Recommended)
1. Go to your dataset on Kaggle
2. Click "Settings" → "Rename"
3. Change to `zuco-preprocessed`

### **Option B: Change the code**
Edit this line in your notebook after copying files:

```python
# Add this cell after copying files
import fileinput
import sys

# Update config.py to use your dataset name
with fileinput.FileInput('config.py', inplace=True) as f:
    for line in f:
        if 'zuco-preprocessed' in line:
            line = line.replace('zuco-preprocessed', 'YOUR-DATASET-NAME')
        print(line, end='')

print('✓ Updated dataset path')
```

---

## ⏱️ Expected Timeline on Kaggle GPU

| Stage | Epochs | GPU Type | Time |
|-------|--------|----------|------|
| Pre-training | 5 | T4 | 3-5 hours |
| Pre-training | 5 | P100 | 2-3 hours |
| Fine-tuning | 30 | T4 | 20-30 hours |
| Fine-tuning | 30 | P100 | 15-20 hours |
| **TOTAL** | **35** | **T4** | **~25-35 hours** |
| **TOTAL** | **35** | **P100** | **~17-23 hours** |

**Note**: Kaggle GPU sessions timeout after **12 hours**. You'll need to resume training across multiple sessions.

---

## 🔄 Resume Training After Timeout

If your session times out, resume from checkpoint:

```python
# Add this in your next Kaggle session
# (Steps 1-4 stay the same)

# CELL 5 (Modified): Resume training
import sys
sys.argv = [
    'train_simple.py',
    '--resume',
    '--checkpoint', '/kaggle/working/models/simple_training/checkpoint_epoch_10.pt'
]
exec(open('train_simple.py').read())
```

Or manually edit `train_simple.py` to load checkpoint:

```python
# Add after line 280 in train_simple.py:
checkpoint_path = save_dir / 'checkpoint_epoch_10.pt'
if checkpoint_path.exists():
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f'✓ Resumed from epoch {start_epoch}')
else:
    start_epoch = 1
```

---

## 🐛 Troubleshooting

### **Error: "No such file: /kaggle/input/zuco-preprocessed"**
**Fix**: Add your dataset in Kaggle UI:
1. Click "Add data" → "Your datasets"
2. Select your `processed_zuco` dataset
3. Make sure it's named `zuco-preprocessed`

### **Error: "CUDA out of memory"**
**Fix**: In Cell 2, after copying files, add:
```python
# Reduce batch size for smaller GPU
with open('config.py', 'r') as f:
    config = f.read()
config = config.replace('BATCH_SIZE = 4', 'BATCH_SIZE = 2')
config = config.replace('BATCH_SIZE = 1', 'BATCH_SIZE = 1')
with open('config.py', 'w') as f:
    f.write(config)
```

### **Error: "Internet must be enabled"**
**Fix**: Enable internet in Kaggle settings:
1. Click "Settings" (right sidebar)
2. Internet → Toggle to "On"
3. This allows downloading pre-trained BART model

---

## 📁 What Gets Saved

All outputs go to `/kaggle/working/models/simple_training/`:

```
/kaggle/working/models/simple_training/
├── pretrained_encoder.pt          # After Stage 1 (5 epochs)
├── best_model.pt                  # Best validation loss
├── checkpoint_epoch_5.pt          # Every 5 epochs
├── checkpoint_epoch_10.pt
├── checkpoint_epoch_15.pt
├── checkpoint_epoch_20.pt
├── checkpoint_epoch_25.pt
└── checkpoint_epoch_30.pt
```

Each checkpoint contains:
- Full model weights
- Optimizer state
- Training metrics
- Epoch number

---

## ✅ Complete Checklist

Before clicking "Run All":

- [ ] GPU enabled (Runtime → Change runtime type → GPU)
- [ ] Internet enabled (Settings → Internet → On)
- [ ] Dataset added (`zuco-preprocessed`)
- [ ] All cells copied from this guide
- [ ] Dataset name matches in code

---

## 🎓 Pro Tips

1. **Use P100 GPU** if available (faster than T4)
2. **Enable "Save & Run All"** to auto-save output
3. **Commit session frequently** to preserve checkpoints
4. **Download checkpoints** after every session
5. **Monitor GPU memory**: Click "Environment" to see usage

---

**That's it! Just copy these cells and run! Your code already works on Kaggle! 🎉**
