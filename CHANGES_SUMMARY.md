# ✅ KAGGLE SETUP - COMPLETE SUMMARY

## 🎯 What Changed in Your Code

### **✅ AUTOMATIC PATH DETECTION ADDED**

Your `config.py` now **automatically detects** Kaggle environment:

```python
# Auto-detect environment (Kaggle vs Local)
IS_KAGGLE = os.path.exists('/kaggle/input')

if IS_KAGGLE:
    # Use Kaggle paths
    PROCESSED_DATA_DIR = Path('/kaggle/input/zuco-preprocessed')
    MODEL_SAVE_DIR = Path('/kaggle/working/models')
    OUTPUT_DIR = Path('/kaggle/working/outputs')
    RESULTS_DIR = Path('/kaggle/working/results')
else:
    # Use local paths
    PROCESSED_DATA_DIR = Path('./processed_zuco')
    MODEL_SAVE_DIR = Path('./models')
    # ... etc
```

**Result**: Code works on **both local and Kaggle** without any manual changes! 🎉

---

## 📦 What You Need to Do

### **1. Upload to Kaggle** (One-time setup):

**Your GitHub Repo**: Already done! ✅
- URL: `https://github.com/ShahnaShaj/exp`
- Contains all necessary files

**Your Dataset**: Upload `processed_zuco/` folder
1. Go to [kaggle.com/datasets](https://www.kaggle.com/datasets)
2. Click "New Dataset"
3. Upload your `processed_zuco/` folder (containing `.pkl` files)
4. Name it: **`zuco-preprocessed`** (important!)
5. Set visibility: Public or Private

---

### **2. Create Kaggle Notebook** and copy these cells:

```python
# ===== CELL 1: Install dependencies =====
!pip install -q transformers>=4.30.0 accelerate>=0.20.0 peft>=0.7.0 datasets sacrebleu rouge-score

# ===== CELL 2: Clone and setup =====
import os, shutil

!git clone https://github.com/ShahnaShaj/exp.git

for f in ['train_simple.py','models_simple.py','dataset.py','config.py','utils.py']:
    shutil.copy2(f'exp/{f}', f'/kaggle/working/{f}')
    print(f'✓ {f}')

os.chdir('/kaggle/working')

# ===== CELL 3: Add dataset in UI =====
# Click "Add data" → "Your datasets" → Select "zuco-preprocessed"
from pathlib import Path
print(f"Data ready: {Path('/kaggle/input/zuco-preprocessed').exists()}")

# ===== CELL 4: Verify GPU =====
import torch
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"}')

# ===== CELL 5: RUN TRAINING 🚀 =====
!python train_simple.py

# ===== CELL 6: Check results =====
!ls -lh /kaggle/working/models/simple_training/
```

---

### **3. Click "Save & Run All"**

Training will run for ~25-35 hours (with timeouts, spread across multiple sessions)

---

## 📂 Files You Must Upload

### **To GitHub** (already done):
```
✅ train_simple.py
✅ models_simple.py
✅ dataset.py
✅ config.py (now with auto-detection!)
✅ utils.py
✅ requirements.txt
```

### **To Kaggle Datasets**:
```
✅ processed_zuco/ folder
   ├── train_subjects.pkl
   ├── val_subjects.pkl
   └── test_subjects.pkl
```

**Dataset name MUST be**: `zuco-preprocessed`

---

## 🔧 Zero Manual Path Changes Required!

Your code now automatically uses:

| Environment | Data Path | Output Path |
|------------|-----------|-------------|
| **Local (Windows)** | `./processed_zuco` | `./models` |
| **Kaggle** | `/kaggle/input/zuco-preprocessed` | `/kaggle/working/models` |

Detection happens in `config.py` line 18:
```python
IS_KAGGLE = os.path.exists('/kaggle/input')
```

---

## 📊 What Gets Saved on Kaggle

Training automatically saves to `/kaggle/working/`:

```
/kaggle/working/
└── models/
    └── simple_training/
        ├── pretrained_encoder.pt      # Stage 1 complete
        ├── best_model.pt              # Best validation loss
        ├── checkpoint_epoch_5.pt      # Every 5 epochs
        ├── checkpoint_epoch_10.pt
        ├── checkpoint_epoch_15.pt
        └── ... (up to epoch 30)
```

**Download**: Click "Save Version" → outputs saved in "Output" tab

---

## ⏱️ Training Timeline

| Stage | Duration | Checkpoints |
|-------|----------|-------------|
| Pre-training (5 epochs) | 3-5 hours | `pretrained_encoder.pt` |
| Fine-tuning (30 epochs) | 20-30 hours | `checkpoint_epoch_X.pt` every 5 epochs |
| **Total** | **~25-35 hours** | **7 checkpoint files** |

**Note**: Kaggle times out after 12 hours → resume from checkpoint in next session

---

## 🎯 Key Features Built-In

✅ **Pre-trained model**: BART downloaded automatically from HuggingFace
✅ **Your dataset**: Uses your uploaded `processed_zuco/` data
✅ **Two-stage training**: Pre-train encoder (5) → Fine-tune (30)
✅ **Auto-checkpointing**: Saves best model + every 5 epochs
✅ **Resume capability**: Load from checkpoint if session times out
✅ **LoRA efficiency**: Only 2M trainable params (0.86% of total)
✅ **Path auto-detection**: Works on local and Kaggle without changes

---

## 🚀 Quick Start Summary

1. **Upload dataset** to Kaggle Datasets (name: `zuco-preprocessed`)
2. **Create Kaggle notebook**, copy 6 cells from above
3. **Add dataset** in UI: "Add data" → "Your datasets" → Select it
4. **Enable GPU**: Settings → Accelerator → GPU
5. **Enable Internet**: Settings → Internet → On
6. **Run all cells** → training starts automatically!

---

## 📚 Reference Documents Created

I created **3 comprehensive guides** for you:

1. **`KAGGLE_QUICK_START.md`** ← START HERE
   - 3-step setup guide
   - Copy-paste ready cells
   - Fastest way to get running

2. **`KAGGLE_PATH_CONFIG.md`**
   - Detailed explanation of all path changes
   - Manual configuration options
   - Troubleshooting section

3. **`KAGGLE_SETUP.md`** (already existed)
   - Complete training guide
   - Expected timelines
   - Requirements and features

---

## ✅ Final Checklist

Before running on Kaggle:

- [x] Code updated with auto-detection (`config.py`)
- [x] GitHub repo has all files (`ShahnaShaj/exp`)
- [ ] Upload `processed_zuco/` to Kaggle Datasets
- [ ] Name dataset: `zuco-preprocessed`
- [ ] Create Kaggle notebook
- [ ] Copy 6 cells from Quick Start guide
- [ ] Enable GPU in Kaggle
- [ ] Enable Internet in Kaggle
- [ ] Add dataset in notebook UI
- [ ] Click "Run All"

---

## 🎉 Result

**Your code is now fully Kaggle-compatible with ZERO manual path edits needed!**

Just:
1. Upload dataset → Kaggle Datasets (`zuco-preprocessed`)
2. Copy 6 cells → Kaggle Notebook
3. Run → Training starts automatically

**Training produces**:
- 7 checkpoint files
- Best model (validation loss)
- Pre-trained encoder
- Full training logs
- Ready for evaluation

---

**Everything is ready! Check `KAGGLE_QUICK_START.md` and start training! 🚀**
