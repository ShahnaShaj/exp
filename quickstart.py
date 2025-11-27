"""
Quick Start Guide Generator
Creates a personalized quick start based on your setup
"""

import sys
from pathlib import Path
from config import Config


def check_environment():
    """Check if environment is properly set up"""
    issues = []
    warnings = []
    
    print("="*80)
    print("EEG2TEXT Environment Check")
    print("="*80)
    
    # Check Python version
    if sys.version_info < (3, 8):
        issues.append("Python 3.8+ required")
    else:
        print(f"✓ Python version: {sys.version_info.major}.{sys.version_info.minor}")
    
    # Check for CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            warnings.append("CUDA not available - training will be slow on CPU")
    except ImportError:
        issues.append("PyTorch not installed")
    
    # Check for ZuCo data
    if not Config.ZUCO_ROOT.exists():
        issues.append(f"ZuCo data directory not found: {Config.ZUCO_ROOT}")
    else:
        mat_files = list(Config.ZUCO_ROOT.glob("**/*.mat"))
        if len(mat_files) == 0:
            issues.append("No .mat files found in ZuCo directory")
        else:
            print(f"✓ Found {len(mat_files)} .mat files in ZuCo directory")
    
    # Check for required packages
    required_packages = [
        'torch', 'transformers', 'scipy', 'numpy', 'pandas',
        'h5py', 'sklearn', 'nltk', 'sacrebleu', 'rouge_score'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        issues.append(f"Missing packages: {', '.join(missing_packages)}")
    else:
        print(f"✓ All required packages installed")
    
    # Check disk space
    import shutil
    stat = shutil.disk_usage('.')
    free_gb = stat.free / (1024**3)
    if free_gb < 10:
        warnings.append(f"Low disk space: {free_gb:.1f} GB available (recommend 20+ GB)")
    else:
        print(f"✓ Disk space available: {free_gb:.1f} GB")
    
    # Print results
    print("\n" + "="*80)
    if issues:
        print("❌ Issues Found:")
        for issue in issues:
            print(f"  • {issue}")
        print("\n⚠ Please fix these issues before continuing")
        return False
    
    if warnings:
        print("⚠ Warnings:")
        for warning in warnings:
            print(f"  • {warning}")
    
    print("\n✅ Environment check passed!")
    return True


def generate_quickstart():
    """Generate personalized quick start commands"""
    print("\n" + "="*80)
    print("Quick Start Commands")
    print("="*80)
    
    print("\n1. Install Dependencies:")
    print("   pip install -r requirements.txt")
    
    print("\n2. Verify ZuCo Dataset Structure:")
    print(f"   Your zuco_data/ should contain:")
    print(f"   • task1-SR/ (with 12 .mat files)")
    print(f"   • task2-NR/ (with 12 .mat files)")
    print(f"   • task3-TSR/ (with 12 .mat files)")
    
    print("\n3. Run Complete Pipeline:")
    print("   python run_pipeline.py")
    
    print("\n   OR run steps individually:")
    print("   python zuco_preprocessor.py  # Preprocess data")
    print("   python pretrain.py           # Pre-training")
    print("   python train.py              # Main training")
    print("   python evaluate.py           # Evaluation")
    
    print("\n4. Monitor Training:")
    print("   tensorboard --logdir outputs/")
    
    print("\n5. View Results:")
    print("   cat results/test/metrics.json")
    print("   cat results/test/predictions.txt")


def estimate_training_time():
    """Estimate training time based on setup"""
    print("\n" + "="*80)
    print("Estimated Training Time")
    print("="*80)
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            if 'A40' in gpu_name or 'A100' in gpu_name:
                print("  With 4× A40/A100 (as in paper):")
                print("    • Pre-training: 6-8 hours")
                print("    • Main training: 20-30 hours")
                print("    • Total: ~30-40 hours")
            elif 'RTX' in gpu_name or '3090' in gpu_name or '4090' in gpu_name:
                print("  With single RTX 3090/4090:")
                print("    • Pre-training: 10-15 hours")
                print("    • Main training: 40-60 hours")
                print("    • Total: ~50-75 hours")
            else:
                print("  With your GPU:")
                print("    • Pre-training: 15-20 hours")
                print("    • Main training: 60-80 hours")
                print("    • Total: ~75-100 hours")
        else:
            print("  With CPU only:")
            print("    ⚠ Not recommended - will take several days")
            print("    • Consider using Google Colab or cloud GPU")
    except:
        print("  Unable to estimate - install PyTorch first")


def main():
    print("\n")
    print("╔════════════════════════════════════════════════════════════════════════════╗")
    print("║                                                                            ║")
    print("║                    EEG2TEXT Setup & Quick Start Guide                     ║")
    print("║                                                                            ║")
    print("║          Open Vocabulary EEG-to-Text Decoding with EEG Pre-Training       ║")
    print("║                                                                            ║")
    print("╚════════════════════════════════════════════════════════════════════════════╝")
    
    # Check environment
    env_ok = check_environment()
    
    if not env_ok:
        print("\n❌ Please fix the issues above before proceeding")
        return
    
    # Generate quick start
    generate_quickstart()
    
    # Estimate training time
    estimate_training_time()
    
    print("\n" + "="*80)
    print("Next Steps")
    print("="*80)
    print("\n1. If environment check passed, you're ready to start!")
    print("2. Run: python run_pipeline.py")
    print("3. Or follow the step-by-step commands above")
    print("\n📚 For detailed information, see README.md")
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
