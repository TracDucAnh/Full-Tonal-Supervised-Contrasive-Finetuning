import os
import warnings

import torch
import torchaudio
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import TSNE

from tqdm import tqdm

warnings.filterwarnings('ignore')

class LexicalSoundDataset(Dataset):
    """Dataset for supervised contrastive learning of lexical tones"""
    
    def __init__(self, dataset_dir, sr=16000, duration=1.0):
        self.dataset_dir = dataset_dir
        self.sr = sr
        self.duration = duration
        self.samples = []
        
        # Folder -> label mapping
        self.tone_map = {
            'sắc': 0,
            'huyền': 1,
            'hỏi': 2,
            'ngã': 3,
            'nặng': 4,
            'không dấu': 5
        }
        
        self._load_samples()
    
    def _is_valid_wav(self, wav_path):
        """Check if WAV file is valid and can be loaded"""
        try:
            # Try to load just the metadata (fast check)
            info = torchaudio.info(wav_path)
            # Additional check: make sure it has valid sample rate and channels
            if info.sample_rate <= 0 or info.num_channels <= 0:
                return False
            return True
        except Exception as e:
            # If any error occurs, the file is invalid
            return False
    
    def _load_samples(self):
        """Load all .wav files from TTS model folders"""
        total_files = 0
        valid_files = 0
        invalid_files = 0
        
        print("\nScanning dataset and checking WAV files...")
        
        # Iterate through TTS model folders (edge-tts, google-tts, etc.)
        for tts_model in os.listdir(self.dataset_dir):
            tts_path = os.path.join(self.dataset_dir, tts_model)
            if not os.path.isdir(tts_path):
                continue
            
            # Iterate through tone folders inside each TTS model
            for tone_name, label in self.tone_map.items():
                tone_dir = os.path.join(tts_path, tone_name)
                if not os.path.exists(tone_dir):
                    continue
                
                # Load all .wav files from tone folder
                for filename in os.listdir(tone_dir):
                    if filename.endswith('.wav'):
                        wav_path = os.path.join(tone_dir, filename)
                        total_files += 1
                        
                        # Check if file is valid before adding
                        if self._is_valid_wav(wav_path):
                            self.samples.append((wav_path, label, tone_name, tts_model))
                            valid_files += 1
                        else:
                            invalid_files += 1
                            if invalid_files <= 10:  # Only print first 10 invalid files
                                print(f"  ⚠ Skipping invalid file: {wav_path}")
        
        print(f"\n{'='*70}")
        print(f"Dataset scan completed:")
        print(f"  Total WAV files found: {total_files}")
        print(f"  Valid files: {valid_files}")
        print(f"  Invalid/corrupted files: {invalid_files}")
        if invalid_files > 10:
            print(f"  (Only first 10 invalid files shown)")
        print(f"{'='*70}\n")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        wav_path, label, tone_name, tts_model = self.samples[idx]
        
        try:
            # Load audio
            waveform, sr = torchaudio.load(wav_path)
            
            # Resample if needed
            if sr != self.sr:
                resampler = torchaudio.transforms.Resample(sr, self.sr)
                waveform = resampler(waveform)
            
            # Normalize length
            target_length = int(self.sr * self.duration)
            if waveform.shape[1] < target_length:
                waveform = torch.nn.functional.pad(waveform, (0, target_length - waveform.shape[1]))
            else:
                waveform = waveform[:, :target_length]
            
            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            return {
                'waveform': waveform.squeeze(0),
                'label': torch.tensor(label, dtype=torch.long),
                'tone_name': tone_name,
                'tts_model': tts_model
            }
        
        except Exception as e:
            # If loading fails, return a silent audio sample
            # This should rarely happen since we pre-check files
            print(f"\n⚠ Error loading file at runtime: {wav_path}")
            print(f"  Error: {e}")
            print(f"  Returning silent audio sample instead\n")
            
            target_length = int(self.sr * self.duration)
            return {
                'waveform': torch.zeros(target_length),
                'label': torch.tensor(label, dtype=torch.long),
                'tone_name': tone_name,
                'tts_model': tts_model
            }


def create_dataloaders(dataset_dir, batch_size=32, num_workers=0, train_split=0.8, use_cuda=False):
    """Create train and validation dataloaders with CUDA optimization"""
    
    dataset = LexicalSoundDataset(dataset_dir)
    
    # Split dataset
    train_size = int(len(dataset) * train_split)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create dataloaders with pin_memory for faster GPU transfer
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_cuda  # Enable pin_memory when using CUDA
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda
    )
    
    return train_loader, val_loader, dataset
