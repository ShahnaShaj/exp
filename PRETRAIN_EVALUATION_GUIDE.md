# 📊 Pre-trained Model Evaluation Guide

## Overview
The `evaluate_pretrain.py` script provides comprehensive analysis of your pre-trained EEG encoder with detailed visualizations and metrics.

## What It Evaluates

### 1. **Reconstruction Quality** 🎯
- **Metrics**: MSE, MAE, Correlation coefficients
- **Visualizations**:
  - Sample reconstructions (original vs reconstructed)
  - Per-channel reconstruction errors
  - Temporal reconstruction errors
  - Correlation distribution histogram

### 2. **Learned Representations** 🧠
- **Analysis**: PCA and t-SNE dimensionality reduction
- **Metrics**: Explained variance ratios
- **Visualizations**:
  - PCA variance plots
  - 2D embedding scatter plots (PCA and t-SNE)
  - Shows how well the encoder captures EEG structure

### 3. **Attention Patterns** 👁️
- **Analysis**: Transformer attention weights across layers
- **Visualizations**:
  - Attention heatmaps showing which timesteps attend to each other
  - Reveals temporal dependencies learned by the model

### 4. **Channel Importance** 🔬
- **Analysis**: Ablation study - zeros out each channel individually
- **Metrics**: Impact on reconstruction loss per channel
- **Visualizations**:
  - Channel importance bar charts
  - Top 20 most important channels
  - Helps identify critical EEG channels

## How to Run

### Basic Usage
```powershell
python evaluate_pretrain.py
```

### Requirements
- Pre-trained model checkpoint at: `models/pretraining/best_pretrain.pt`
- Preprocessed data files
- ~2-4GB RAM available
- 5-15 minutes runtime (depends on which analyses are enabled)

## Output Files

All results are saved to: `results/pretrain_evaluation/`

### Metrics File
- `pretrain_metrics.json`: All numerical metrics in JSON format

### Report File
- `evaluation_report.txt`: Human-readable summary

### Visualizations (PNG files)
1. `reconstruction_samples.png` - Side-by-side original/reconstructed/error
2. `channel_errors.png` - Bar chart of per-channel MSE
3. `temporal_errors.png` - Time-series of reconstruction quality
4. `correlation_distribution.png` - Histogram of correlations
5. `pca_variance.png` - PCA explained variance plots
6. `embeddings_pca.png` - 2D PCA projection of embeddings
7. `embeddings_t-sne.png` - 2D t-SNE projection of embeddings
8. `attention_heatmap.png` - Attention pattern visualization
9. `channel_importance.png` - Channel ablation results

## Key Metrics to Look For

### Good Pre-training Indicators ✅

1. **Reconstruction Correlation** > 0.6
   - Shows model captures EEG patterns accurately
   
2. **Low MSE/MAE**
   - MSE < 0.5 is good
   - Lower is better
   
3. **PCA Variance**
   - First 10 components should explain >50% variance
   - Shows learned representations are informative
   
4. **Consistent Channel Errors**
   - No single channel with extremely high error
   - All channels reconstructed reasonably well

### Warning Signs ⚠️

1. **Low Correlation** < 0.3
   - Model not learning meaningful patterns
   - May need more training or different hyperparameters
   
2. **High Channel-Specific Errors**
   - Some channels might be noisy or problematic
   - Consider excluding outlier channels
   
3. **Poor PCA Variance**
   - First 50 components explain <70% variance
   - Embeddings may not be informative

## Customization

### Speed vs Completeness Trade-off

The script has a section for channel importance analysis that's commented out by default because it's slow:

```python
# evaluator.evaluate_channel_importance(test_loader)  # Uncomment for full analysis
```

**Uncomment this line** if you want the full channel ablation study (adds ~10-20 minutes).

### Adjust Sample Sizes

In the code, you can modify:
- `max_samples=1000` for representation analysis (line ~100)
- `embeddings[:500]` for t-SNE (line ~150)
- Number of batches for attention analysis (line ~170)

### Change Batch Size

```python
evaluator = PretrainEvaluator(checkpoint_path, device='cpu')
```

Smaller batches = less memory but slower.

## Interpreting Results

### Example Good Results
```
Reconstruction Loss: 0.234
MSE: 0.421
MAE: 0.156
Correlation: 0.682 ± 0.089
PCA variance (50 components): 78.3%
```

### Example Needs Improvement
```
Reconstruction Loss: 1.234
MSE: 2.341
MAE: 0.892
Correlation: 0.234 ± 0.198
PCA variance (50 components): 42.1%
```

## Next Steps After Evaluation

### If Results Are Good ✅
1. Proceed to Stage 2 training: `python train_stage2.py`
2. Use the pre-trained encoder for downstream tasks
3. Consider freezing encoder weights in early stage2 epochs

### If Results Need Improvement ⚠️
1. **Continue Pre-training**: Run more epochs
2. **Adjust Hyperparameters**:
   - Increase model capacity (hidden_dim)
   - Change mask ratio (try 0.20 or 0.25)
   - Adjust learning rate
3. **Data Quality**: Check for noisy channels or subjects
4. **Different Architecture**: Try different num_layers or num_heads

## Troubleshooting

### "Checkpoint not found" Error
- Make sure you've run `python pretrain.py` first
- Check the path: `models/pretraining/best_pretrain.pt`

### Memory Errors
- Reduce batch_size in the evaluation script
- Disable channel importance analysis (already default)
- Reduce max_samples for representation analysis

### "No module named matplotlib"
```powershell
pip install matplotlib seaborn scikit-learn
```

### Slow t-SNE
- Normal! t-SNE is computationally expensive
- It processes only 500 samples by default
- You can reduce this if needed

## Technical Details

### What Each Analysis Tests

**Reconstruction Quality**: Tests if the encoder preserves information when faced with missing data (masks). High correlation means good information retention.

**PCA Analysis**: Checks if learned representations are structured and informative. High explained variance in few components = efficient encoding.

**t-SNE Clustering**: Visualizes if similar EEG patterns cluster together in embedding space. Good clusters = meaningful learned features.

**Attention Patterns**: Shows what temporal relationships the model learned. Diagonal patterns = local dependencies, off-diagonal = long-range dependencies.

**Channel Importance**: Identifies which EEG sensors contribute most to reconstruction. Helps understand brain regions most relevant for the task.

## Integration with Training Pipeline

This evaluation helps you decide:
1. **When to stop pre-training**: Good metrics = ready for supervised training
2. **Which checkpoints to use**: Compare multiple checkpoints
3. **Hyperparameter tuning**: Quantify effects of different settings
4. **Debug issues**: Identify problems before expensive supervised training

---

**Recommended Workflow:**

```powershell
# 1. Pre-train model
python pretrain.py

# 2. Evaluate pre-training
python evaluate_pretrain.py

# 3. Review results
cat results/pretrain_evaluation/evaluation_report.txt

# 4. If good, proceed to supervised training
python train_stage2.py
```

---

Good luck with your evaluation! 🚀
