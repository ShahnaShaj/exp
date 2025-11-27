"""
Combine individual task files into all_data_processed.pkl
"""

import pickle
import json
from pathlib import Path
from config import Config

def combine_task_files():
    """Load individual task files and combine them"""
    
    processed_dir = Path(Config.PROCESSED_DATA_DIR)
    
    if not processed_dir.exists():
        print(f"Error: Processed data directory not found: {processed_dir}")
        return
    
    print("="*70)
    print("Combining Task Files")
    print("="*70)
    
    all_data = []
    tasks = ['task1-SR', 'task2-NR', 'task3-TSR']
    
    # Load each task
    for task in tasks:
        task_file = processed_dir / f"{task}_processed.pkl"
        
        if not task_file.exists():
            print(f"⚠ Warning: {task_file.name} not found, skipping...")
            continue
        
        print(f"\nLoading {task_file.name}...")
        with open(task_file, 'rb') as f:
            task_data = pickle.load(f)
        
        print(f"  ✓ Loaded {len(task_data)} sentences")
        all_data.extend(task_data)
    
    if not all_data:
        print("\n❌ No data loaded!")
        return
    
    print(f"\n{'='*70}")
    print(f"Total: {len(all_data)} sentences")
    print("="*70)
    
    # Save combined file
    combined_file = processed_dir / "all_data_processed.pkl"
    print(f"\nSaving to {combined_file}...")
    
    with open(combined_file, 'wb') as f:
        pickle.dump(all_data, f)
    
    print(f"✓ Saved successfully!")
    print(f"\nFile location: {combined_file}")
    print(f"File size: {combined_file.stat().st_size / (1024**2):.1f} MB")
    
    print("\n" + "="*70)
    print("✅ Combining completed!")
    print("="*70)


if __name__ == '__main__':
    combine_task_files()
