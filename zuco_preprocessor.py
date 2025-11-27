"""
ZuCo Dataset Preprocessor
Handles loading and preprocessing of ZuCo EEG data from .mat files
Organizes data by tasks: task1-SR, task2-NR, task3-TSR
"""

import numpy as np
import scipy.io
import h5py
from pathlib import Path
import pickle
import json
from tqdm import tqdm
from config import Config


class ZuCoPreprocessor:
    """Preprocess ZuCo dataset with proper folder structure"""
    
    def __init__(self, zuco_root_path=None):
        """
        Args:
            zuco_root_path: Path to folder containing task1-SR, task2-NR, task3-TSR
        """
        self.zuco_root = Path(zuco_root_path or Config.ZUCO_ROOT)
        self.tasks = Config.TASKS
        self.subject_codes = Config.SUBJECT_CODES
        self.num_channels = Config.NUM_CHANNELS
        
        if not self.zuco_root.exists():
            raise FileNotFoundError(f"ZuCo root path not found: {self.zuco_root}")
    
    def explore_mat_structure(self, mat_file_path):
        """
        Explore the structure of a .mat file
        Useful for understanding the data format
        """
        print(f"\n{'='*70}")
        print(f"Exploring: {mat_file_path.name}")
        print(f"{'='*70}")
        
        try:
            # Try scipy first (for MATLAB v7 and earlier)
            data = scipy.io.loadmat(
                str(mat_file_path), 
                struct_as_record=False, 
                squeeze_me=True
            )
            print("✓ Loaded with scipy.io.loadmat")
            
            print("\nTop-level keys:")
            for key in data.keys():
                if not key.startswith('__'):
                    print(f"  - {key}: {type(data[key])}")
                    if hasattr(data[key], 'shape'):
                        print(f"    Shape: {data[key].shape}")
            
            # Inspect sentenceData structure
            if 'sentenceData' in data:
                sent_data = data['sentenceData']
                print(f"\nsentenceData type: {type(sent_data)}")
                
                if isinstance(sent_data, np.ndarray):
                    print(f"Number of sentences: {len(sent_data)}")
                    
                    if len(sent_data) > 0:
                        first_sent = sent_data[0] if sent_data.ndim > 0 else sent_data
                        
                        print(f"\nFirst sentence fields:")
                        if hasattr(first_sent, '_fieldnames'):
                            for field in first_sent._fieldnames:
                                print(f"  - {field}")
                        
                        # Check word structure
                        if hasattr(first_sent, 'word'):
                            words = first_sent.word
                            num_words = len(words) if isinstance(words, np.ndarray) else 1
                            print(f"\nNumber of words in first sentence: {num_words}")
                            
                            first_word = words[0] if isinstance(words, np.ndarray) else words
                            if hasattr(first_word, '_fieldnames'):
                                print(f"\nWord fields:")
                                for field in first_word._fieldnames:
                                    if hasattr(first_word, field):
                                        val = getattr(first_word, field)
                                        shape_info = f", shape: {val.shape}" if hasattr(val, 'shape') else ""
                                        print(f"  - {field}: {type(val).__name__}{shape_info}")
            
            return data
            
        except NotImplementedError as e:
            print(f"scipy failed (likely MATLAB v7.3 format): {e}")
            print("\nTrying h5py for MATLAB v7.3 format...")
            
            try:
                with h5py.File(str(mat_file_path), 'r') as f:
                    print("✓ Loaded with h5py")
                    print("\nTop-level keys:")
                    for key in f.keys():
                        print(f"  - {key}: {type(f[key])}")
                        if isinstance(f[key], h5py.Dataset):
                            print(f"    Shape: {f[key].shape}")
                return None
            except Exception as e2:
                print(f"h5py also failed: {e2}")
                return None
        
        except Exception as e:
            print(f"Error loading file: {e}")
            return None
    
    def load_sentence_data(self, mat_file_path, verbose=False):
        """
        Load sentence-level continuous EEG data from a ZuCo .mat file
        Extracts raw EEG signal for the entire sentence reading period
        
        Returns:
            list of dicts with keys: 'eeg', 'text', 'sentence_idx'
        """
        try:
            # Load mat file
            data = scipy.io.loadmat(
                str(mat_file_path), 
                struct_as_record=False, 
                squeeze_me=True
            )
            
            if 'sentenceData' not in data:
                if verbose:
                    print(f"    Warning: 'sentenceData' not found in {mat_file_path.name}")
                return []
            
            sentence_data = data['sentenceData']
            
            # Ensure it's iterable
            if not isinstance(sentence_data, np.ndarray):
                sentence_data = [sentence_data]
            
            processed_sentences = []
            
            for sent_idx, sent_struct in enumerate(sentence_data):
                try:
                    # Extract sentence text
                    if not hasattr(sent_struct, 'content'):
                        continue
                    
                    sentence_text = sent_struct.content
                    if isinstance(sentence_text, np.ndarray):
                        sentence_text = str(sentence_text[0]) if len(sentence_text) > 0 else ""
                    else:
                        sentence_text = str(sentence_text)
                    
                    # Skip empty sentences
                    if not sentence_text or len(sentence_text.strip()) == 0:
                        continue
                    
                    # Extract raw continuous EEG for the entire sentence
                    # ZuCo stores raw EEG in 'rawData' or similar field
                    sentence_eeg = None
                    
                    # Try to get raw EEG data from various possible fields
                    if hasattr(sent_struct, 'rawData'):
                        sentence_eeg = sent_struct.rawData
                    elif hasattr(sent_struct, 'eeg'):
                        sentence_eeg = sent_struct.eeg
                    elif hasattr(sent_struct, 'omit'):
                        # Sometimes raw data is in omit field
                        sentence_eeg = sent_struct.omit
                    
                    # If no direct sentence-level EEG, concatenate word-level raw data
                    if sentence_eeg is None or (isinstance(sentence_eeg, np.ndarray) and sentence_eeg.size == 0):
                        if hasattr(sent_struct, 'word'):
                            word_data = sent_struct.word
                            if not isinstance(word_data, np.ndarray):
                                word_data = [word_data]
                            
                            # Collect raw EEG from each word and concatenate
                            word_eeg_segments = []
                            for word_struct in word_data:
                                if hasattr(word_struct, 'rawData'):
                                    raw_eeg = word_struct.rawData
                                    if isinstance(raw_eeg, np.ndarray) and raw_eeg.size > 0:
                                        # Ensure shape is (time_steps, channels)
                                        if raw_eeg.ndim == 1:
                                            raw_eeg = raw_eeg.reshape(-1, 1)
                                        elif raw_eeg.shape[0] < raw_eeg.shape[1]:
                                            # If channels are in first dim, transpose
                                            raw_eeg = raw_eeg.T
                                        word_eeg_segments.append(raw_eeg)
                            
                            if word_eeg_segments:
                                # Concatenate all word segments along time axis
                                sentence_eeg = np.concatenate(word_eeg_segments, axis=0)
                    
                    # Process the EEG data
                    if sentence_eeg is not None:
                        if isinstance(sentence_eeg, np.ndarray) and sentence_eeg.size > 0:
                            # Ensure correct shape: (time_steps, channels)
                            if sentence_eeg.ndim == 1:
                                sentence_eeg = sentence_eeg.reshape(-1, 1)
                            elif sentence_eeg.shape[0] < sentence_eeg.shape[1]:
                                # If channels are in first dim, transpose
                                sentence_eeg = sentence_eeg.T
                            
                            # Ensure we have the right number of channels
                            if sentence_eeg.shape[1] != self.num_channels:
                                if sentence_eeg.shape[1] > self.num_channels:
                                    # Take first num_channels
                                    sentence_eeg = sentence_eeg[:, :self.num_channels]
                                else:
                                    # Pad with zeros
                                    padding = np.zeros((sentence_eeg.shape[0], 
                                                      self.num_channels - sentence_eeg.shape[1]))
                                    sentence_eeg = np.hstack([sentence_eeg, padding])
                            
                            processed_sentences.append({
                                'eeg': sentence_eeg,  # Shape: (time_steps, 105)
                                'text': sentence_text,
                                'sentence_idx': sent_idx
                            })
                
                except Exception as e:
                    if verbose:
                        print(f"    Warning: Could not process sentence {sent_idx}: {e}")
                    continue
            
            return processed_sentences
            
        except Exception as e:
            if verbose:
                print(f"  Error loading {mat_file_path.name}: {e}")
            return []
    
    def normalize_eeg(self, eeg_data, target_channels=None):
        """
        Normalize continuous EEG data using z-score normalization
        
        Args:
            eeg_data: numpy array of shape (time_steps, channels)
            target_channels: target number of channels (default: Config.NUM_CHANNELS)
        
        Returns:
            Normalized EEG array of shape (time_steps, channels)
        """
        if target_channels is None:
            target_channels = self.num_channels
        
        # Ensure correct number of channels
        if eeg_data.shape[1] != target_channels:
            if eeg_data.shape[1] > target_channels:
                eeg_data = eeg_data[:, :target_channels]
            else:
                padding = np.zeros((eeg_data.shape[0], target_channels - eeg_data.shape[1]))
                eeg_data = np.hstack([eeg_data, padding])
        
        # Z-score normalization per channel across time
        mean = np.mean(eeg_data, axis=0, keepdims=True)
        std = np.std(eeg_data, axis=0, keepdims=True)
        
        # Avoid division by zero
        std[std == 0] = 1.0
        
        normalized = (eeg_data - mean) / std
        
        # Clip extreme values to prevent outliers
        normalized = np.clip(normalized, -10, 10)
        
        return normalized
    
    def process_all_data(self, output_dir=None, explore_first=True):
        """
        Process all tasks and subjects
        
        Args:
            output_dir: Directory to save processed data
            explore_first: Whether to explore the first file structure
        
        Returns:
            tuple: (all_data, statistics)
        """
        output_path = Path(output_dir or Config.PROCESSED_DATA_DIR)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Explore first file structure if requested
        if explore_first:
            for task in self.tasks:
                task_path = self.zuco_root / task
                if task_path.exists():
                    mat_files = list(task_path.glob('*.mat'))
                    if mat_files:
                        self.explore_mat_structure(mat_files[0])
                        break
            
            print("\n" + "="*70)
            response = input("Continue with full preprocessing? (y/n): ")
            if response.lower() != 'y':
                print("Preprocessing cancelled.")
                return [], {}
        
        all_data = []
        statistics = {
            'total_sentences': 0,
            'total_words': 0,
            'by_task': {},
            'by_subject': {},
            'avg_eeg_length': [],
            'avg_sentence_length': []
        }
        
        print("\n" + "="*70)
        print("Starting Full Preprocessing")
        print("="*70)
        
        for task in self.tasks:
            task_path = self.zuco_root / task
            
            if not task_path.exists():
                print(f"\n⚠ Warning: Task folder not found: {task_path}")
                continue
            
            print(f"\n{'='*70}")
            print(f"Processing Task: {task}")
            print(f"{'='*70}")
            
            task_data = []
            
            # Find all .mat files in this task folder
            mat_files = sorted(list(task_path.glob('*.mat')))
            print(f"Found {len(mat_files)} .mat files")
            
            for mat_file in tqdm(mat_files, desc=f"Processing {task}"):
                # Load sentences from this file
                sentences = self.load_sentence_data(mat_file, verbose=False)
                
                if sentences:
                    # Normalize EEG data
                    for sent in sentences:
                        sent['eeg'] = self.normalize_eeg(sent['eeg'])
                        sent['task'] = task
                        sent['subject_file'] = mat_file.name
                        
                        # Extract subject code from filename
                        # Format: resultsZAB_SR.mat -> ZAB
                        subject_code = mat_file.stem.replace('results', '').split('_')[0]
                        sent['subject'] = subject_code
                        
                        # Track statistics
                        statistics['avg_eeg_length'].append(sent['eeg'].shape[0])
                        statistics['avg_sentence_length'].append(len(sent['text'].split()))
                    
                    task_data.extend(sentences)
                    all_data.extend(sentences)
                    
                    # Update statistics
                    statistics['total_sentences'] += len(sentences)
                    statistics['total_words'] += sum(len(s['text'].split()) for s in sentences)
                    
                    if task not in statistics['by_task']:
                        statistics['by_task'][task] = 0
                    statistics['by_task'][task] += len(sentences)
                    
                    subject_name = mat_file.stem
                    if subject_name not in statistics['by_subject']:
                        statistics['by_subject'][subject_name] = 0
                    statistics['by_subject'][subject_name] += len(sentences)
            
            # Save task-specific data
            if task_data:
                task_output_file = output_path / f"{task}_processed.pkl"
                with open(task_output_file, 'wb') as f:
                    pickle.dump(task_data, f)
                print(f"  ✓ Saved {len(task_data)} sentences to {task_output_file.name}")
        
        # Calculate average statistics
        if statistics['avg_eeg_length']:
            statistics['avg_eeg_length'] = float(np.mean(statistics['avg_eeg_length']))
            statistics['avg_sentence_length'] = float(np.mean(statistics['avg_sentence_length']))
        
        # Save all data combined
        if all_data:
            combined_file = output_path / "all_data_processed.pkl"
            with open(combined_file, 'wb') as f:
                pickle.dump(all_data, f)
            
            # Save statistics
            stats_file = output_path / "statistics.json"
            with open(stats_file, 'w') as f:
                json.dump(statistics, f, indent=2)
            
            # Print summary
            self._print_summary(statistics, output_path)
            
            return all_data, statistics
        else:
            print("\n⚠ Warning: No data processed!")
            return [], statistics
    
    def _print_summary(self, statistics, output_path):
        """Print processing summary"""
        print(f"\n{'='*70}")
        print("PREPROCESSING SUMMARY")
        print(f"{'='*70}")
        print(f"✓ Total sentences: {statistics['total_sentences']}")
        print(f"✓ Total words: {statistics['total_words']}")
        print(f"✓ Average EEG length: {statistics['avg_eeg_length']:.1f} time steps")
        print(f"✓ Average sentence length: {statistics['avg_sentence_length']:.1f} words")
        
        print(f"\nBy task:")
        for task, count in statistics['by_task'].items():
            print(f"  • {task}: {count} sentences")
        
        print(f"\nBy subject:")
        for subject, count in sorted(statistics['by_subject'].items()):
            print(f"  • {subject}: {count} sentences")
        
        print(f"\n✓ Processed data saved to: {output_path}/")
        print(f"  • all_data_processed.pkl")
        print(f"  • statistics.json")
        for task in statistics['by_task'].keys():
            print(f"  • {task}_processed.pkl")
        print("="*70)


def main():
    """Main preprocessing function"""
    print("="*70)
    print("ZuCo Dataset Preprocessing Pipeline")
    print("="*70)
    
    # Initialize preprocessor
    preprocessor = ZuCoPreprocessor()
    
    # Process all data
    all_data, stats = preprocessor.process_all_data(
        output_dir=Config.PROCESSED_DATA_DIR,
        explore_first=True
    )
    
    if all_data:
        print("\n✅ Preprocessing completed successfully!")
    else:
        print("\n❌ Preprocessing failed. Check errors above.")


if __name__ == '__main__':
    main()
