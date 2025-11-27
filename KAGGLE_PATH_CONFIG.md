# 🔧 Kaggle Path Configuration Guide

## Quick Summary: What You Need to Change

When running on Kaggle, you need to update paths in **2 files**:
1. **`config.py`** - All base paths
2. **Optional**: Command-line arguments when running scripts

---

## 📂 Kaggle Directory Structure

Kaggle notebooks have this structure:
```
/kaggle/
├── input/                    # READ-ONLY (your uploaded datasets)
│   ├── your-repo-name/      # Your cloned GitHub repo
│   │   ├── train_simple.py
│   │   ├── models_simple.py
│   │   ├── config.py
│   │   └── ...
│   └── zuco-preprocessed/   # Your preprocessed data (if uploaded as dataset)
│       ├── train_subjects.pkl
│       ├── val_subjects.pkl
│       └── test_subjects.pkl
│
└── working/                  # READ-WRITE (your outputs)
    ├── models/              # Saved model checkpoints
    ├── outputs/             # Training logs
    └── results/             # Evaluation results
```

---

## ✏️ STEP 1: Edit `config.py`

Open `config.py` and change these lines:

### **BEFORE (Windows/Local)**:
```python
class Config:
    # Data paths
    ZUCO_ROOT = Path('./zuco_data')
    PROCESSED_DATA_DIR = Path('./processed_zuco')
    OUTPUT_DIR = Path('./outputs')
    MODEL_SAVE_DIR = Path('./models')
    RESULTS_DIR = Path('./results')
```

### **AFTER (Kaggle)**:
```python
class Config:
    # Data paths - KAGGLE VERSION
    ZUCO_ROOT = Path('/kaggle/input/zuco-raw')  # If you uploaded raw data
    PROCESSED_DATA_DIR = Path('/kaggle/input/zuco-preprocessed')  # Your preprocessed data
    OUTPUT_DIR = Path('/kaggle/working/outputs')
    MODEL_SAVE_DIR = Path('/kaggle/working/models')
    RESULTS_DIR = Path('/kaggle/working/results')
```

**Important Notes**:
- `/kaggle/input/` is **READ-ONLY** - you can't write here
- `/kaggle/working/` is **READ-WRITE** - all outputs go here
- Change `zuco-preprocessed` to match your Kaggle dataset name

---

## 🚀 STEP 2: Copy Code Files to Working Directory

Since `/kaggle/input/` is read-only, copy your code to `/kaggle/working/`:

### **Add this cell at the start of your Kaggle notebook**:

```python
# Cell 1: Copy repo files to working directory
import shutil
import os

# Copy all Python files from input to working directory
repo_path = '/kaggle/input/your-repo-name'  # Change to your repo name
working_path = '/kaggle/working'

# List of files to copy
files_to_copy = [
    'train_simple.py',
    'models_simple.py',
    'dataset.py',
    'config.py',
    'utils.py',
    'train_advanced.py',  # Optional
    'models_advanced.py',  # Optional
    'evaluate.py',        # Optional
]

for file in files_to_copy:
    src = f'{repo_path}/{file}'
    dst = f'{working_path}/{file}'
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f'✓ Copied {file}')
    else:
        print(f'⚠ {file} not found')

# Change to working directory
os.chdir('/kaggle/working')
print('\n✓ Working directory:', os.getcwd())
```

---

## 📝 STEP 3: Update config.py in Working Directory

After copying, update the paths:

```python
# Cell 2: Update config.py for Kaggle paths
config_content = """
# Data paths - KAGGLE VERSION
ZUCO_ROOT = Path('/kaggle/input/zuco-raw')
PROCESSED_DATA_DIR = Path('/kaggle/input/zuco-preprocessed')
OUTPUT_DIR = Path('/kaggle/working/outputs')
MODEL_SAVE_DIR = Path('/kaggle/working/models')
RESULTS_DIR = Path('/kaggle/working/results')
"""

# Read current config
with open('config.py', 'r') as f:
    config = f.read()

# Replace paths section
import re
pattern = r"ZUCO_ROOT = Path\('.*?'\).*?RESULTS_DIR = Path\('.*?'\)"
replacement = """ZUCO_ROOT = Path('/kaggle/input/zuco-raw')
    PROCESSED_DATA_DIR = Path('/kaggle/input/zuco-preprocessed')
    OUTPUT_DIR = Path('/kaggle/working/outputs')
    MODEL_SAVE_DIR = Path('/kaggle/working/models')
    RESULTS_DIR = Path('/kaggle/working/results')"""

config = re.sub(pattern, replacement, config, flags=re.DOTALL)

# Write updated config
with open('config.py', 'w') as f:
    f.write(config)

print('✓ Updated config.py with Kaggle paths')
```

---

## 🎯 STEP 4: Run Training

Now just run normally:

```python
# Cell 3: Run training
!python train_simple.py
```

All outputs will be saved to `/kaggle/working/models/` and `/kaggle/working/outputs/`

---

## 📦 Alternative: Manual Path Changes

If you prefer to manually edit `config.py` before uploading to GitHub:

### **Option A: Environment Variable Detection (Recommended)**

Add this to the **top of `config.py`**:

```python
import os
from pathlib import Path

# Auto-detect Kaggle environment
IS_KAGGLE = os.path.exists('/kaggle/input')

class Config:
    # Data paths - Auto-detect Kaggle or Local
    if IS_KAGGLE:
        ZUCO_ROOT = Path('/kaggle/input/zuco-raw')
        PROCESSED_DATA_DIR = Path('/kaggle/input/zuco-preprocessed')
        OUTPUT_DIR = Path('/kaggle/working/outputs')
        MODEL_SAVE_DIR = Path('/kaggle/working/models')
        RESULTS_DIR = Path('/kaggle/working/results')
    else:
        # Local paths (Windows/Mac/Linux)
        ZUCO_ROOT = Path('./zuco_data')
        PROCESSED_DATA_DIR = Path('./processed_zuco')
        OUTPUT_DIR = Path('./outputs')
        MODEL_SAVE_DIR = Path('./models')
        RESULTS_DIR = Path('./results')
```

**This way, the code works on both local and Kaggle without any changes!**

---

## 🗂️ Dataset Upload Names

Make sure your Kaggle dataset names match:

### **If you uploaded preprocessed data**:
```python
PROCESSED_DATA_DIR = Path('/kaggle/input/zuco-preprocessed')
# or whatever you named it in Kaggle Datasets
```

### **If you uploaded raw ZuCo data**:
```python
ZUCO_ROOT = Path('/kaggle/input/zuco-eeg-raw')
```

To find your dataset path in Kaggle:
1. In notebook, click "Add data" → "Your datasets"
2. Select your dataset
3. Path appears as `/kaggle/input/your-dataset-name`

---

## 📊 Where Your Outputs Will Be Saved

After training, your files will be at:

```
/kaggle/working/
├── models/
│   └── simple_training/
│       ├── pretrained_encoder.pt     # Stage 1 checkpoint
│       ├── best_model.pt             # Best model
│       ├── checkpoint_epoch_5.pt     # Periodic checkpoint
│       ├── checkpoint_epoch_10.pt
│       └── ...
│
├── outputs/
│   └── training_logs.txt
│
└── results/
    └── evaluation_metrics.json
```

**To download these**:
1. Click "Save Version" → "Save & Run All"
2. Wait for completion
3. Go to "Output" tab
4. Download the entire `/kaggle/working/` folder

---

## 🔍 Verification Checklist

Before running training, verify paths:

```python
# Cell: Verify all paths
import os
from pathlib import Path

# Check data input
data_path = Path('/kaggle/input/zuco-preprocessed')
print(f"Data exists: {data_path.exists()}")
if data_path.exists():
    print(f"Files: {list(data_path.glob('*.pkl'))}")

# Check working directory
print(f"\nWorking dir: {os.getcwd()}")

# Check output directories exist
output_dirs = [
    '/kaggle/working/models',
    '/kaggle/working/outputs',
    '/kaggle/working/results'
]
for d in output_dirs:
    Path(d).mkdir(exist_ok=True, parents=True)
    print(f"✓ {d} ready")

# Check required files
required_files = [
    'train_simple.py',
    'models_simple.py',
    'config.py',
    'dataset.py',
    'utils.py'
]
for f in required_files:
    exists = Path(f).exists()
    print(f"{'✓' if exists else '✗'} {f}")
```

---

## 🚨 Common Path Errors and Fixes

### **Error: `FileNotFoundError: ./processed_zuco`**
**Cause**: config.py still has local paths
**Fix**: Update `PROCESSED_DATA_DIR` to `/kaggle/input/zuco-preprocessed`

### **Error: `OSError: Read-only file system`**
**Cause**: Trying to write to `/kaggle/input/`
**Fix**: Change `MODEL_SAVE_DIR` to `/kaggle/working/models`

### **Error: `No such file or directory: models/simple_training`**
**Cause**: Output directories not created
**Fix**: Add this at start of training script:
```python
Path('/kaggle/working/models').mkdir(exist_ok=True, parents=True)
```

### **Error: `ModuleNotFoundError: No module named 'config'`**
**Cause**: Not in the right directory
**Fix**: Run `os.chdir('/kaggle/working')` first

---

## 📋 Complete Kaggle Notebook Template

Here's a complete ready-to-use notebook:

```python
# =============================================================================
# CELL 1: Setup environment
# =============================================================================
import os
import shutil
from pathlib import Path

# Copy repo files to working directory
repo_path = '/kaggle/input/neurolens'  # YOUR REPO NAME HERE
files = [
    'train_simple.py', 'models_simple.py', 'dataset.py',
    'config.py', 'utils.py'
]

for file in files:
    src = f'{repo_path}/{file}'
    if os.path.exists(src):
        shutil.copy2(src, '/kaggle/working/')
        print(f'✓ {file}')

os.chdir('/kaggle/working')

# =============================================================================
# CELL 2: Update config.py paths
# =============================================================================
import re

with open('config.py', 'r') as f:
    config = f.read()

# Replace paths
replacements = {
    "Path('./zuco_data')": "Path('/kaggle/input/zuco-raw')",
    "Path('./processed_zuco')": "Path('/kaggle/input/zuco-preprocessed')",
    "Path('./outputs')": "Path('/kaggle/working/outputs')",
    "Path('./models')": "Path('/kaggle/working/models')",
    "Path('./results')": "Path('/kaggle/working/results')"
}

for old, new in replacements.items():
    config = config.replace(old, new)

with open('config.py', 'w') as f:
    f.write(config)

print('✓ Config updated for Kaggle')

# =============================================================================
# CELL 3: Install dependencies
# =============================================================================
!pip install -q transformers accelerate peft datasets sacrebleu rouge-score

# =============================================================================
# CELL 4: Verify setup
# =============================================================================
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"}')
print(f'\nData path exists: {Path("/kaggle/input/zuco-preprocessed").exists()}')

# =============================================================================
# CELL 5: Run training
# =============================================================================
!python train_simple.py

# =============================================================================
# CELL 6: Check outputs
# =============================================================================
!ls -lh /kaggle/working/models/simple_training/
```

---

## ✅ Final Summary

### **MUST CHANGE in config.py**:
```python
PROCESSED_DATA_DIR = Path('/kaggle/input/zuco-preprocessed')  # Input data
MODEL_SAVE_DIR = Path('/kaggle/working/models')               # Output checkpoints
OUTPUT_DIR = Path('/kaggle/working/outputs')                  # Output logs
RESULTS_DIR = Path('/kaggle/working/results')                 # Output results
```

### **Dataset name must match**:
- Upload your `processed_zuco/` folder to Kaggle Datasets
- Name it something like `zuco-preprocessed`
- Use that name in the path: `/kaggle/input/YOUR-DATASET-NAME`

### **No other changes needed**:
- All other code references `Config.PROCESSED_DATA_DIR`, etc.
- Training scripts will use the updated paths automatically

---

**That's it! Change those 4 paths in config.py and you're ready to train on Kaggle! 🚀**
