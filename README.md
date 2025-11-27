# EEG2TEXT: Open Vocabulary EEG-to-Text Decoding

Complete implementation of the EEG2TEXT model from the paper: **"Open Vocabulary EEG-to-Text Decoding with EEG Pre-Training"**

This implementation supports the ZuCo dataset with proper handling of the task1-SR, task2-NR, and task3-TSR folder structure.

---

## 📁 Project Structure

```
neurolens/
├── config.py                  # Configuration and hyperparameters
├── models.py                  # Model architectures (CNN, Transformer, Multi-View, EEG2TEXT)
├── dataset.py                 # Dataset classes and data loaders
├── utils.py                   # Utility functions
├── zuco_preprocessor.py       # ZuCo data preprocessing
├── pretrain.py                # Self-supervised pre-training
├── train.py                   # Supervised training
├── evaluate.py                # Model evaluation with BLEU/ROUGE
├── run_pipeline.py            # Complete pipeline runner
├── requirements.txt           # Python dependencies
│
├── zuco_data/                 # Raw ZuCo dataset (you provide)
│   ├── task1-SR/
│   │   ├── resultsZAB_SR.mat
│   │   ├── resultsZDM_SR.mat
│   │   └── ... (12 subjects)
│   ├── task2-NR/
│   │   └── ... (12 subjects)
│   └── task3-TSR/
│       └── ... (12 subjects)
│
├── processed_zuco/            # Preprocessed data (generated)
│   ├── all_data_processed.pkl
│   ├── statistics.json
│   └── task*_processed.pkl
│
├── models/                    # Saved model checkpoints (generated)
│   ├── pretraining/
│   │   └── best_pretrain.pt
│   └── main_training/
│       └── best_model.pt
│
├── results/                   # Evaluation results (generated)
│   ├── validation/
│   │   ├── metrics.json
│   │   └── predictions.txt
│   └── test/
│       ├── metrics.json
│       └── predictions.txt
│
└── outputs/                   # Training logs (generated)
    ├── pretrain_logs/
    └── train_logs/
```

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Install Python dependencies
pip install -r requirements.txt
```

### 2. Prepare ZuCo Dataset

Place your ZuCo dataset in the `zuco_data/` directory with the following structure:

```
zuco_data/
├── task1-SR/
│   ├── resultsZAB_SR.mat
│   ├── resultsZDM_SR.mat
│   └── ... (12 subjects total)
├── task2-NR/
│   └── ... (12 subjects)
└── task3-TSR/
    └── ... (12 subjects)
```

**Download ZuCo:** [https://osf.io/q3zws/](https://osf.io/q3zws/)

### 3. Run Complete Pipeline

**Option A: Run everything at once**
```bash
python run_pipeline.py
```

**Option B: Run step-by-step**

```bash
# Step 1: Preprocess data
python zuco_preprocessor.py

# Step 2: Pre-training (self-supervised)
python pretrain.py

# Step 3: Main training (supervised)
python train.py

# Step 4: Evaluation
python evaluate.py
```

---

## 📊 Model Architecture

### Components

1. **CNN Compressor**: Reduces EEG sequence length by 4x
2. **Transformer Encoder**: Extracts features from compressed EEG
3. **Multi-View Transformer**: Separate encoders for 10 brain regions
4. **BART Decoder**: Pre-trained language model for text generation

### Brain Regions (10 regions from paper Table 1)

- Prefrontal
- Premotor
- Broca's area
- Auditory association
- Primary motor
- Primary sensory
- Somatic sensory
- Auditory
- Wernicke's area
- Visual

---

## ⚙️ Configuration

Edit `config.py` to customize hyperparameters:

```python
# Model architecture
HIDDEN_DIM = 512
NUM_TRANSFORMER_LAYERS = 6
NUM_ATTENTION_HEADS = 8

# Pre-training
PRETRAIN_EPOCHS = 10
PRETRAIN_BATCH_SIZE = 8
PRETRAIN_LR = 5e-5

# Main training
TRAIN_EPOCHS = 50
BATCH_SIZE = 4
LEARNING_RATE = 3e-5
```

---

## 📈 Training Details

### Pre-training Phase (Self-Supervised)

- **Task**: Masked EEG reconstruction
- **Masking**: 15% of EEG segments randomly masked
- **Strategy**: Re-randomize masks each epoch (best performance)
- **Duration**: ~10 epochs
- **Loss**: MSE between reconstructed and original EEG

### Main Training Phase (Supervised)

- **Task**: EEG-to-Text generation
- **Encoder**: Multi-view transformer (pre-trained)
- **Decoder**: BART (facebook/bart-base)
- **Duration**: ~50 epochs
- **Loss**: Cross-entropy on generated text

---

## 📊 Evaluation Metrics

The model is evaluated using:

- **BLEU-1, BLEU-2, BLEU-3, BLEU-4**: N-gram overlap
- **ROUGE-1, ROUGE-2, ROUGE-L**: Recall-oriented scores
- **Word Accuracy**: Word-level correctness

### Expected Results (from paper)

| Metric | Paper Result |
|--------|--------------|
| BLEU-1 | 0.452 |
| BLEU-4 | 0.141 |
| ROUGE-1 | 0.342 |

---

## 🔍 Monitoring Training

### TensorBoard

```bash
# View pre-training logs
tensorboard --logdir outputs/pretrain_logs

# View main training logs
tensorboard --logdir outputs/train_logs
```

### Check Results

```bash
# View metrics
cat results/test/metrics.json

# View sample predictions
cat results/test/predictions.txt
```

---

## 💾 Saved Models

Models are saved in `models/` directory:

- `pretraining/best_pretrain.pt`: Best pre-trained encoder
- `main_training/best_model.pt`: Best EEG2TEXT model
- `main_training/model_epoch_*.pt`: Periodic checkpoints

### Loading Trained Models

```python
from models import EEG2TEXT
from config import Config
import torch

# Initialize model
model = EEG2TEXT(
    channel_groups=Config.CHANNEL_GROUPS,
    hidden_dim=Config.HIDDEN_DIM
)

# Load checkpoint
checkpoint = torch.load('models/main_training/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Generate predictions
model.eval()
with torch.no_grad():
    outputs = model.generate(eeg_input, max_length=50, num_beams=5)
```

---

## 🧪 Testing Individual Components

Test each module independently:

```bash
# Test models
python models.py

# Test datasets
python dataset.py

# Test utilities
python utils.py
```

---

## 📝 Preprocessing Details

The preprocessing script (`zuco_preprocessor.py`):

1. Loads .mat files from task folders
2. Extracts sentence-level EEG and text
3. Normalizes EEG (z-score per channel)
4. Pads/truncates to 105 channels
5. Saves to pickle files

**Output:**
- `all_data_processed.pkl`: All data combined
- `statistics.json`: Dataset statistics
- Individual task files

---

## 🔧 Troubleshooting

### Issue: Out of Memory

**Solution 1**: Reduce batch size in `config.py`
```python
BATCH_SIZE = 2  # Default: 4
PRETRAIN_BATCH_SIZE = 4  # Default: 8
```

**Solution 2**: Reduce max EEG length
```python
MAX_EEG_LENGTH = 1500  # Default: 2000
```

**Solution 3**: Use gradient accumulation
```python
# In train.py, accumulate gradients over multiple batches
```

### Issue: CUDA out of memory

```bash
# Use CPU
export CUDA_VISIBLE_DEVICES=""
```

Or in `config.py`:
```python
DEVICE = torch.device('cpu')
```

### Issue: Slow data loading

```python
# 12
NUM_WORKERS = 0  # Default: 4
```

### Issue: Pre-trained weights not found

```bash
# Run pre-training first
python pretrain.py

# Or train from scratch (skip pre-training)
# The model will warn but continue with random initialization
```

---

## 📊 Hardware Requirements

### Minimum (Training will be slow)
- CPU: 4+ cores
- RAM: 16 GB
- GPU: None (CPU only)
- Storage: 10 GB

### Recommended (from paper)
- CPU: 16+ cores
- RAM: 64 GB
- GPU: 4× NVIDIA A40 (40GB each)
- Storage: 50 GB

### Practical (Single GPU)
- CPU: 8+ cores
- RAM: 32 GB
- GPU: 1× NVIDIA RTX 3090/4090 (24GB)
- Storage: 20 GB

---

## 📚 Paper Reference

```bibtex
@article{wang2024eeg2text,
  title={Open Vocabulary EEG-to-Text Decoding with EEG Pre-Training},
  author={Wang, et al.},
  journal={arXiv preprint},
  year={2024}
}
```

---

## 🛠️ Advanced Usage

### Training on Specific Task

```python
# In dataset.py
from dataset import load_preprocessed_data

# Load only Task 1 (Sentence Reading)
data = load_preprocessed_data(task='task1-SR')
```

### Custom Hyperparameters

```python
# Create custom config
class CustomConfig(Config):
    HIDDEN_DIM = 768
    NUM_TRANSFORMER_LAYERS = 12
    LEARNING_RATE = 1e-5
```

### Resume Training

```python
# In train.py main()
trainer.train(
    ...
    resume_from='models/main_training/model_epoch_10.pt'
)
```

---

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review the paper for implementation details
3. Check model outputs and logs

---

## ✅ Checklist

Before training:
- [ ] ZuCo dataset downloaded and placed in `zuco_data/`
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Sufficient disk space (~20 GB)
- [ ] GPU available (recommended) or prepared for slow CPU training

After training:
- [ ] Check `results/test/metrics.json` for performance
- [ ] Compare with paper results
- [ ] Review sample predictions in `results/test/predictions.txt`

---

## 🎯 Next Steps

1. **Improve Performance**:
   - Train for more epochs
   - Tune hyperparameters
   - Use larger BART model (`facebook/bart-large`)

2. **Experiment**:
   - Try different masking strategies
   - Adjust learning rates
   - Test on individual tasks

3. **Deploy**:
   - Export model for inference
   - Create real-time EEG processing pipeline
   - Build interactive demo

---

**Happy Training! 🧠→📝**
