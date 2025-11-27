"""
Evaluation script for EEG2TEXT
Computes BLEU, ROUGE, and other metrics
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU only


import torch
from tqdm import tqdm
import numpy as np
from pathlib import Path
from transformers import BartTokenizer
from sacrebleu import corpus_bleu
from rouge_score import rouge_scorer
import json

from config import Config
from models import EEG2TEXT
from dataset import load_preprocessed_data, split_data, create_dataloaders
from utils import set_seed, get_device_info, load_checkpoint


class Evaluator:
    """Evaluator for EEG2TEXT model"""
    
    def __init__(self, model, tokenizer, device):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
    
    def generate_predictions(self, dataloader, max_length=50, num_beams=5):
        """
        Generate predictions for all samples in dataloader
        
        Args:
            dataloader: Data loader
            max_length: Maximum generation length
            num_beams: Number of beams for beam search
        
        Returns:
            predictions: List of predicted texts
            references: List of reference texts
        """
        predictions = []
        references = []
        
        print("\nGenerating predictions...")
        with torch.no_grad():
            for batch in tqdm(dataloader):
                eeg = batch['eeg'].to(self.device)
                # Handle both 'text' and 'texts' keys for compatibility
                true_texts = batch.get('texts', batch.get('text', []))
                # Ensure true_texts is always a list
                if isinstance(true_texts, str):
                    true_texts = [true_texts]
                elif not isinstance(true_texts, list):
                    true_texts = list(true_texts)
                
                # Generate predictions
                outputs = self.model.generate(
                    eeg,
                    max_length=max_length,
                    num_beams=num_beams
                )
                
                # Decode predictions
                pred_texts = [
                    self.tokenizer.decode(ids, skip_special_tokens=True)
                    for ids in outputs
                ]
                
                predictions.extend(pred_texts)
                references.extend(true_texts)
        
        return predictions, references
    
    def compute_bleu(self, predictions, references):
        """
        Compute BLEU scores (BLEU-1, BLEU-2, BLEU-3, BLEU-4)
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts
        
        Returns:
            dict with BLEU scores
        """
        # sacrebleu expects references as a list of lists
        # sacrebleu expects references as list of lists and predictions as list
        refs = [[ref] for ref in references]
        
        # Handle empty predictions/references
        if not predictions or not references:
            return {'BLEU-1': 0.0, 'BLEU-2': 0.0, 'BLEU-3': 0.0, 'BLEU-4': 0.0}
        
        # Compute BLEU once - it returns all n-gram precisions
        bleu = corpus_bleu(predictions, refs, force=True, lowercase=True)
        
        # Extract individual n-gram precisions and overall BLEU-4 score
        bleu_scores = {
            'BLEU-1': bleu.precisions[0] / 100.0,
            'BLEU-2': bleu.precisions[1] / 100.0,
            'BLEU-3': bleu.precisions[2] / 100.0,
            'BLEU-4': bleu.score / 100.0  # Overall BLEU-4 score with geometric mean
        }
        
        return bleu_scores
    
    def compute_rouge(self, predictions, references):
        """
        Compute ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts
        
        Returns:
            dict with ROUGE scores
        """
        # Handle empty predictions/references
        if not predictions or not references:
            return {
                'ROUGE-1-F': 0.0, 'ROUGE-1-P': 0.0, 'ROUGE-1-R': 0.0,
                'ROUGE-2-F': 0.0, 'ROUGE-2-P': 0.0, 'ROUGE-2-R': 0.0,
                'ROUGE-L-F': 0.0, 'ROUGE-L-P': 0.0, 'ROUGE-L-R': 0.0
            }
        
        scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )
        
        rouge_scores = {
            'rouge1_f': [],
            'rouge1_p': [],
            'rouge1_r': [],
            'rouge2_f': [],
            'rouge2_p': [],
            'rouge2_r': [],
            'rougeL_f': [],
            'rougeL_p': [],
            'rougeL_r': []
        }
        
        for pred, ref in zip(predictions, references):
            scores = scorer.score(ref, pred)
            
            rouge_scores['rouge1_f'].append(scores['rouge1'].fmeasure)
            rouge_scores['rouge1_p'].append(scores['rouge1'].precision)
            rouge_scores['rouge1_r'].append(scores['rouge1'].recall)
            
            rouge_scores['rouge2_f'].append(scores['rouge2'].fmeasure)
            rouge_scores['rouge2_p'].append(scores['rouge2'].precision)
            rouge_scores['rouge2_r'].append(scores['rouge2'].recall)
            
            rouge_scores['rougeL_f'].append(scores['rougeL'].fmeasure)
            rouge_scores['rougeL_p'].append(scores['rougeL'].precision)
            rouge_scores['rougeL_r'].append(scores['rougeL'].recall)
        
        # Average scores (handle empty lists)
        avg_rouge = {
            'ROUGE-1-F': np.mean(rouge_scores['rouge1_f']) if rouge_scores['rouge1_f'] else 0.0,
            'ROUGE-1-P': np.mean(rouge_scores['rouge1_p']) if rouge_scores['rouge1_p'] else 0.0,
            'ROUGE-1-R': np.mean(rouge_scores['rouge1_r']) if rouge_scores['rouge1_r'] else 0.0,
            'ROUGE-2-F': np.mean(rouge_scores['rouge2_f']) if rouge_scores['rouge2_f'] else 0.0,
            'ROUGE-2-P': np.mean(rouge_scores['rouge2_p']) if rouge_scores['rouge2_p'] else 0.0,
            'ROUGE-2-R': np.mean(rouge_scores['rouge2_r']) if rouge_scores['rouge2_r'] else 0.0,
            'ROUGE-L-F': np.mean(rouge_scores['rougeL_f']) if rouge_scores['rougeL_f'] else 0.0,
            'ROUGE-L-P': np.mean(rouge_scores['rougeL_p']) if rouge_scores['rougeL_p'] else 0.0,
            'ROUGE-L-R': np.mean(rouge_scores['rougeL_r']) if rouge_scores['rougeL_r'] else 0.0
        }
        
        return avg_rouge
    
    def compute_word_accuracy(self, predictions, references):
        """
        Compute word-level accuracy
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts
        
        Returns:
            Word accuracy
        """
        total_correct = 0
        total_words = 0
        
        for pred, ref in zip(predictions, references):
            pred_words = pred.lower().split()
            ref_words = ref.lower().split()
            
            # Count correct words (order-independent)
            for word in pred_words:
                if word in ref_words:
                    total_correct += 1
            
            total_words += len(ref_words)
        
        accuracy = total_correct / total_words if total_words > 0 else 0
        return accuracy
    
    def evaluate(self, dataloader, max_length=50, num_beams=5):
        """
        Complete evaluation with all metrics
        
        Args:
            dataloader: Data loader
            max_length: Maximum generation length
            num_beams: Number of beams for beam search
        
        Returns:
            dict with all evaluation metrics
        """
        # Generate predictions
        predictions, references = self.generate_predictions(
            dataloader, max_length, num_beams
        )
        
        print("\nComputing metrics...")
        
        # Compute BLEU scores
        bleu_scores = self.compute_bleu(predictions, references)
        
        # Compute ROUGE scores
        rouge_scores = self.compute_rouge(predictions, references)
        
        # Compute word accuracy
        word_acc = self.compute_word_accuracy(predictions, references)
        
        # Combine all metrics
        metrics = {
            **bleu_scores,
            **rouge_scores,
            'word_accuracy': word_acc,
            'num_samples': len(predictions)
        }
        
        return metrics, predictions, references


def print_metrics(metrics):
    """Print metrics in a formatted way"""
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    
    print("\nBLEU Scores:")
    for key in ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4']:
        if key in metrics:
            print(f"  {key}: {metrics[key]:.4f}")
    
    print("\nROUGE-1 Scores:")
    for key in ['ROUGE-1-F', 'ROUGE-1-P', 'ROUGE-1-R']:
        if key in metrics:
            print(f"  {key}: {metrics[key]:.4f}")
    
    print("\nROUGE-2 Scores:")
    for key in ['ROUGE-2-F', 'ROUGE-2-P', 'ROUGE-2-R']:
        if key in metrics:
            print(f"  {key}: {metrics[key]:.4f}")
    
    print("\nROUGE-L Scores:")
    for key in ['ROUGE-L-F', 'ROUGE-L-P', 'ROUGE-L-R']:
        if key in metrics:
            print(f"  {key}: {metrics[key]:.4f}")
    
    print(f"\nWord Accuracy: {metrics.get('word_accuracy', 0):.4f}")
    print(f"Number of Samples: {metrics.get('num_samples', 0)}")
    print("="*70)


def print_sample_predictions(predictions, references, num_samples=10):
    """Print sample predictions"""
    print("\n" + "="*70)
    print("SAMPLE PREDICTIONS")
    print("="*70)
    
    for i in range(min(num_samples, len(predictions))):
        print(f"\nSample {i+1}:")
        print(f"  Reference: {references[i]}")
        print(f"  Prediction: {predictions[i]}")
        print("-"*70)


def save_results(metrics, predictions, references, output_path):
    """Save evaluation results to files"""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save metrics as JSON
    metrics_file = output_path / 'metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✓ Metrics saved to: {metrics_file}")
    
    # Save predictions and references
    results_file = output_path / 'predictions.txt'
    with open(results_file, 'w', encoding='utf-8') as f:
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            f.write(f"Sample {i+1}\n")
            f.write(f"Reference: {ref}\n")
            f.write(f"Prediction: {pred}\n")
            f.write("-"*70 + "\n")
    print(f"✓ Predictions saved to: {results_file}")


def main():
    """Main evaluation function"""
    
    # Set seed for reproducibility
    set_seed(Config.SEED)
    
    # Get device
    device = get_device_info()
    
    # Load tokenizer
    print("\n" + "="*70)
    print("LOADING TOKENIZER")
    print("="*70)
    tokenizer = BartTokenizer.from_pretrained(Config.BART_MODEL)
    print(f"✓ Loaded tokenizer: {Config.BART_MODEL}")
    
    # Load preprocessed data
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    
    all_data = load_preprocessed_data()
    
    # Split data
    train_data, val_data, test_data = split_data(
        all_data,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=Config.SEED
    )
    
    # Create dataloaders with batch_size=1 for evaluation (less memory)
    _, val_loader, test_loader = create_dataloaders(
        train_data,
        val_data,
        test_data,
        tokenizer=tokenizer,
        batch_size=1,  # Use batch_size=1 for evaluation to save memory
        num_workers=Config.NUM_WORKERS,
        for_pretraining=False
    )
    
    print(f"✓ Data loaded successfully")
    
    # Initialize model
    print("\n" + "="*70)
    print("LOADING MODEL")
    print("="*70)
    
    model = EEG2TEXT(
        channel_groups=Config.CHANNEL_GROUPS,
        hidden_dim=Config.HIDDEN_DIM,
        num_layers=Config.NUM_TRANSFORMER_LAYERS,
        num_heads=Config.NUM_ATTENTION_HEADS,
        ff_dim=Config.FEEDFORWARD_DIM,
        dropout=Config.DROPOUT,
        bart_model=Config.BART_MODEL
    )
    
    # Load trained model checkpoint
    checkpoint_path = Config.MODEL_SAVE_DIR / 'main_training' / 'best_model.pt'
    
    if not checkpoint_path.exists():
        print(f"\n❌ Error: Model checkpoint not found at {checkpoint_path}")
        print("Please train the model first using: python train.py")
        return
    
    print(f"Loading model from: {checkpoint_path}")
    epoch, val_loss = load_checkpoint(checkpoint_path, model)
    print(f"✓ Model loaded (epoch {epoch}, val_loss: {val_loss:.4f})")
    
    # Initialize evaluator
    evaluator = Evaluator(model, tokenizer, device)
    
    # Evaluate on validation set
    print("\n" + "="*70)
    print("EVALUATING ON VALIDATION SET")
    print("="*70)
    
    val_metrics, val_predictions, val_references = evaluator.evaluate(
        val_loader,
        max_length=Config.MAX_GENERATION_LENGTH,
        num_beams=1  # Use greedy decoding instead of beam search (much less memory)
    )
    
    print_metrics(val_metrics)
    print_sample_predictions(val_predictions, val_references, num_samples=5)
    
    # Save validation results
    save_results(
        val_metrics,
        val_predictions,
        val_references,
        Config.RESULTS_DIR / 'validation'
    )
    
    # Evaluate on test set
    print("\n" + "="*70)
    print("EVALUATING ON TEST SET")
    print("="*70)
    
    test_metrics, test_predictions, test_references = evaluator.evaluate(
        test_loader,
        max_length=Config.MAX_GENERATION_LENGTH,
        num_beams=1  # Use greedy decoding instead of beam search (much less memory)
    )
    
    print_metrics(test_metrics)
    print_sample_predictions(test_predictions, test_references, num_samples=5)
    
    # Save test results
    save_results(
        test_metrics,
        test_predictions,
        test_references,
        Config.RESULTS_DIR / 'test'
    )
    
    # Compare with paper results
    print("\n" + "="*70)
    print("COMPARISON WITH PAPER RESULTS")
    print("="*70)
    print("\nPaper Results (from Table 3):")
    print("  BLEU-1: 0.452")
    print("  BLEU-4: 0.141")
    print("  ROUGE-1: 0.342")
    
    print("\nYour Results (Test Set):")
    print(f"  BLEU-1: {test_metrics.get('BLEU-1', 0):.3f}")
    print(f"  BLEU-4: {test_metrics.get('BLEU-4', 0):.3f}")
    print(f"  ROUGE-1-F: {test_metrics.get('ROUGE-1-F', 0):.3f}")
    print("="*70)
    
    print("\n✅ Evaluation completed!")
    print(f"\nResults saved to: {Config.RESULTS_DIR}")


if __name__ == '__main__':
    main()
