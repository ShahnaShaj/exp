"""
Quick runner script for complete EEG2TEXT pipeline
Run all steps: preprocessing -> pre-training -> training -> evaluation
"""

import subprocess
import sys
from pathlib import Path
from config import Config


def run_command(command, description):
    """Run a command and handle errors"""
    print("\n" + "="*80)
    print(f"{description}")
    print("="*80)
    print(f"Command: {command}")
    print()
    
    result = subprocess.run(command, shell=True)
    
    if result.returncode != 0:
        print(f"\n❌ Error: {description} failed with return code {result.returncode}")
        sys.exit(result.returncode)
    
    print(f"\n✅ {description} completed successfully!")
    return result.returncode


def main():
    print("="*80)
    print("EEG2TEXT Complete Pipeline Runner")
    print("="*80)
    print("\nThis script will run the complete pipeline:")
    print("  1. Data Preprocessing")
    print("  2. Model Pre-training (Self-supervised)")
    print("  3. Main Training (Supervised)")
    print("  4. Evaluation")
    print("\n⚠ Warning: This may take several hours to complete!")
    
    response = input("\nContinue? (y/n): ")
    if response.lower() != 'y':
        print("Pipeline cancelled.")
        return
    
    # Step 1: Preprocessing
    run_command(
        "python zuco_preprocessor.py",
        "Step 1/4: Data Preprocessing"
    )
    
    # Step 2: Pre-training
    run_command(
        "python pretrain.py",
        "Step 2/4: Self-supervised Pre-training"
    )
    
    # Step 3: Main Training
    run_command(
        "python train.py",
        "Step 3/4: Supervised Training"
    )
    
    # Step 4: Evaluation
    run_command(
        "python evaluate.py",
        "Step 4/4: Model Evaluation"
    )
    
    print("\n" + "="*80)
    print("✅ COMPLETE PIPELINE FINISHED SUCCESSFULLY!")
    print("="*80)
    print(f"\nResults saved to:")
    print(f"  • Models: {Config.MODEL_SAVE_DIR}")
    print(f"  • Results: {Config.RESULTS_DIR}")
    print(f"  • Logs: {Config.OUTPUT_DIR}")
    print("\nView TensorBoard logs:")
    print(f"  tensorboard --logdir {Config.OUTPUT_DIR}")


if __name__ == '__main__':
    main()
