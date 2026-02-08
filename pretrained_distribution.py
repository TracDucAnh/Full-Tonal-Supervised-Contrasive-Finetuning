import os
import warnings

import torch
import torchaudio
import numpy as np
import pandas as pd

from transformers import Wav2Vec2Model, Wav2Vec2Processor, HubertModel

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import TSNE

from tqdm import tqdm

warnings.filterwarnings('ignore')

from dataset import LexicalSoundDataset, create_dataloaders

# ============================================================================
# Configuration
# ============================================================================
OUTPUT_DIR = 'pretrained_distribution_graphs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set device with detailed info
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"🚀 CUDA is available!")
    print(f"   Device: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    device = torch.device('cpu')
    print(f"⚠️  CUDA not available, using CPU")

print(f"Using device: {device}")

# Tone mapping and colors
TONE_NAMES = ['sắc', 'huyền', 'hỏi', 'ngã', 'nặng', 'không dấu']

# English names for tones with diacritic names
TONE_ENGLISH_NAMES = {
    'sắc': 'Rising tone',
    'huyền': 'Falling tone',
    'hỏi': 'Dipping tone',
    'ngã': 'Creaky rising tone',
    'nặng': 'Low glottalized tone',
    'không dấu': 'Level tone'
}

TONE_COLORS = {
    'sắc': '#e74c3c',        # Red
    'huyền': '#2ecc71',      # Green
    'hỏi': '#3498db',        # Blue
    'ngã': '#f39c12',        # Orange
    'nặng': '#9b59b6',       # Purple
    'không dấu': '#1abc9c'   # Teal
}


# ============================================================================
# Feature Extractors with CUDA Support
# ============================================================================
class Wav2Vec2FeatureExtractor:
    """Extract features using pretrained Wav2Vec2 model with CUDA support"""
    
    def __init__(self, model_name='facebook/wav2vec2-base', device='cpu'):
        print(f"\nLoading Wav2Vec2 model: {model_name}")
        self.device = device
        self.model_name = model_name
        
        # Load processor and model
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        
        if device.type == 'cuda':
            self.model = self.model
        
        print("✓ Wav2Vec2 model loaded successfully!")
    
    def extract_features(self, waveforms, sample_rate=16000):
        """Extract features from audio waveforms"""
        with torch.no_grad():
            # Convert to numpy if tensor
            if isinstance(waveforms, torch.Tensor):
                waveforms = waveforms.cpu().numpy()
            
            # Process waveforms
            inputs = self.processor(
                waveforms,
                sampling_rate=sample_rate,
                return_tensors="pt",
                padding=True
            )
            
            # Move to device
            input_values = inputs.input_values.to(self.device)
            
            if self.device.type == 'cuda':
                input_values = input_values
            
            # Forward pass
            outputs = self.model(input_values)
            
            # Get last hidden state and average pool over time
            hidden_states = outputs.last_hidden_state
            features = hidden_states.mean(dim=1)  # [batch_size, hidden_dim]
            
            # Convert back to float32 for compatibility
            if self.device.type == 'cuda':
                features = features.float()
            
            return features.cpu().numpy()


class HuBERTFeatureExtractor:
    """Extract features using pretrained HuBERT model with CUDA support"""
    
    def __init__(self, model_name='facebook/hubert-base-ls960', device='cpu'):
        print(f"\nLoading HuBERT model: {model_name}")
        self.device = device
        self.model_name = model_name
        
        # HuBERT uses Wav2Vec2FeatureExtractor (not Wav2Vec2Processor)
        from transformers import Wav2Vec2FeatureExtractor
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = HubertModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        
        if device.type == 'cuda':
            self.model = self.model

        print("✓ HuBERT model loaded successfully!")
    
    def extract_features(self, waveforms, sample_rate=16000):
        """Extract features from audio waveforms"""
        with torch.no_grad():
            # Convert to numpy if tensor
            if isinstance(waveforms, torch.Tensor):
                waveforms = waveforms.cpu().numpy()
            
            # Process waveforms
            inputs = self.processor(
                waveforms,
                sampling_rate=sample_rate,
                return_tensors="pt",
                padding=True
            )
            
            # Move to device
            input_values = inputs.input_values.to(self.device)
            
            if self.device.type == 'cuda':
                input_values = input_values
            
            # Forward pass
            outputs = self.model(input_values)
            
            # Get last hidden state and average pool over time
            hidden_states = outputs.last_hidden_state
            features = hidden_states.mean(dim=1)  # [batch_size, hidden_dim]
            
            # Convert back to float32 for compatibility
            if self.device.type == 'cuda':
                features = features.float()
            
            return features.cpu().numpy()


# ============================================================================
# Feature Extraction from DataLoader
# ============================================================================
def extract_all_features(dataloader, extractor, max_batches=None):
    """
    Extract features from all samples in the dataloader with GPU acceleration
    
    Args:
        dataloader: PyTorch DataLoader
        extractor: Feature extractor instance (Wav2Vec2 or HuBERT)
        max_batches: Maximum number of batches to process (None = all)
    
    Returns:
        features: NumPy array of shape [n_samples, feature_dim]
        labels: NumPy array of label indices
        tone_names: List of tone names
        tts_models: List of TTS model names
    """
    all_features = []
    all_labels = []
    all_tone_names = []
    all_tts_models = []
    
    print(f"\nExtracting features from dataloader...")
    if max_batches:
        print(f"Processing first {max_batches} batches only")
    
    batch_count = 0
    
    # Show GPU memory if available
    if torch.cuda.is_available():
        print(f"Initial GPU memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    
    for batch in tqdm(dataloader, desc="Processing batches"):
        if max_batches and batch_count >= max_batches:
            break
        
        # Extract batch data
        waveforms = batch['waveform']  # [batch_size, audio_length]
        labels = batch['label']        # [batch_size]
        tone_names = batch['tone_name']
        tts_models = batch['tts_model']
        
        # Extract features (automatically uses GPU if available)
        features = extractor.extract_features(waveforms)
        
        all_features.append(features)
        all_labels.extend(labels.numpy())
        all_tone_names.extend(tone_names)
        all_tts_models.extend(tts_models)
        
        batch_count += 1
        
        # Clear GPU cache periodically
        if torch.cuda.is_available() and batch_count % 10 == 0:
            torch.cuda.empty_cache()
    
    # Show final GPU memory
    if torch.cuda.is_available():
        print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
        torch.cuda.reset_peak_memory_stats()
    
    # Concatenate all features
    all_features = np.vstack(all_features)
    all_labels = np.array(all_labels)
    
    print(f"✓ Extracted features shape: {all_features.shape}")
    print(f"✓ Total samples: {len(all_labels)}")
    
    return all_features, all_labels, all_tone_names, all_tts_models


# ============================================================================
# t-SNE Visualization
# ============================================================================
def visualize_tsne(features, labels, tone_names_list, save_path, model_name=''):
    """Visualize features using t-SNE dimensionality reduction"""
    print(f"\nRunning t-SNE for {model_name}...")
    
    # Apply t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=min(30, len(features) // 5),
        n_iter=1000,
        random_state=42,
        verbose=0
    )
    
    features_2d = tsne.fit_transform(features)
    
    # Create plot
    plt.figure(figsize=(14, 10))
    
    # Plot each tone
    for tone_idx, tone_name in enumerate(TONE_NAMES):
        mask = labels == tone_idx
        if mask.sum() == 0:
            continue
        
        color = TONE_COLORS[tone_name]
        english_name = TONE_ENGLISH_NAMES[tone_name]
        
        plt.scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            c=color,
            label=english_name,
            alpha=0.6,
            s=40,
            edgecolors='white',
            linewidth=0.5
        )
    
    title = f'Vietnamese Tone Distribution - t-SNE Visualization\n{model_name}'
    plt.title(title, fontsize=20, fontweight='bold', pad=20)
    plt.xlabel('t-SNE Dimension 1', fontsize=18)
    plt.ylabel('t-SNE Dimension 2', fontsize=18)
    plt.legend(title='Tone', fontsize=16, title_fontsize=16,
               loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ t-SNE plot saved to: {save_path}")
    plt.close()
    
    return features_2d


def print_statistics(features, labels, model_name=''):
    """Print feature statistics by tone"""
    print("\n" + "=" * 70)
    print(f"FEATURE STATISTICS BY TONE - {model_name}")
    print("=" * 70)
    
    for tone_idx, tone_name in enumerate(TONE_NAMES):
        mask = labels == tone_idx
        if mask.sum() == 0:
            continue
        
        tone_features = features[mask]
        
        print(f"\n{tone_name:12s}:")
        print(f"  Samples:      {tone_features.shape[0]:6d}")
        print(f"  Feature mean: {tone_features.mean():8.4f}")
        print(f"  Feature std:  {tone_features.std():8.4f}")
        print(f"  Feature min:  {tone_features.min():8.4f}")
        print(f"  Feature max:  {tone_features.max():8.4f}")
    
    print("\n" + "=" * 70)


def save_results(features, labels, tone_names, tts_models, tsne_coords, model_name=''):
    """Save extracted features and coordinates to CSV"""
    print(f"\nSaving results for {model_name}...")
    
    # Create safe filename
    safe_model_name = model_name.replace('/', '_').replace(' ', '_').lower()
    
    df = pd.DataFrame({
        'tone_label': labels,
        'tone_name': tone_names,
        'tts_model': tts_models,
        'tsne_x': tsne_coords[:, 0],
        'tsne_y': tsne_coords[:, 1]
    })
    
    csv_path = os.path.join(OUTPUT_DIR, f'feature_coordinates_{safe_model_name}.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"✓ Coordinates saved to: {csv_path}")
    
    # Save raw features
    features_path = os.path.join(OUTPUT_DIR, f'features_{safe_model_name}.npy')
    np.save(features_path, features)
    print(f"✓ Features saved to: {features_path}")


# ============================================================================
# Main Visualization Function
# ============================================================================
def visualize_pretrained_features(dataloader, max_batches=16):
    """
    Main function to extract and visualize features from both models
    
    Args:
        dataloader: PyTorch DataLoader with audio samples
        max_batches: Maximum number of batches to process (None = all, default = 16)
    """
    print("=" * 70)
    print("VIETNAMESE TONE FEATURE DISTRIBUTION ANALYSIS")
    print("=" * 70)
    
    # ========================================================================
    # 1. Wav2Vec2 Analysis
    # ========================================================================
    print("\n" + "=" * 70)
    print("ANALYZING WITH WAV2VEC2")
    print("=" * 70)
    
    wav2vec_extractor = Wav2Vec2FeatureExtractor(
        model_name='facebook/wav2vec2-base',
        device=device
    )
    
    wav2vec_features, labels, tone_names, tts_models = extract_all_features(
        dataloader, wav2vec_extractor, max_batches
    )
    
    print_statistics(wav2vec_features, labels, 'Wav2Vec2')
    
    wav2vec_tsne = visualize_tsne(
        wav2vec_features, labels, tone_names,
        os.path.join(OUTPUT_DIR, 'tsne_wav2vec2.png'),
        'Wav2Vec2 Features'
    )
    
    save_results(wav2vec_features, labels, tone_names, tts_models, 
                 wav2vec_tsne, 'wav2vec2')
    
    # Clean up GPU memory
    del wav2vec_extractor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"✓ GPU memory cleared")
    
    # ========================================================================
    # 2. HuBERT Analysis
    # ========================================================================
    print("\n" + "=" * 70)
    print("ANALYZING WITH HUBERT")
    print("=" * 70)
    
    hubert_extractor = HuBERTFeatureExtractor(
        model_name='facebook/hubert-base-ls960',
        device=device
    )
    
    hubert_features, labels, tone_names, tts_models = extract_all_features(
        dataloader, hubert_extractor, max_batches
    )
    
    print_statistics(hubert_features, labels, 'HuBERT')
    
    hubert_tsne = visualize_tsne(
        hubert_features, labels, tone_names,
        os.path.join(OUTPUT_DIR, 'tsne_hubert.png'),
        'HuBERT Features'
    )
    
    save_results(hubert_features, labels, tone_names, tts_models, 
                 hubert_tsne, 'hubert')
    
    # Clean up GPU memory
    del hubert_extractor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"✓ GPU memory cleared")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print(f"\n{'=' * 70}")
    print("ANALYSIS COMPLETED!")
    print("=" * 70)
    print(f"All visualizations saved to: {OUTPUT_DIR}/")
    print(f"\nGenerated files:")
    print(f"  Wav2Vec2:")
    print(f"    - tsne_wav2vec2.png")
    print(f"    - feature_coordinates_wav2vec2.csv")
    print(f"    - features_wav2vec2.npy")
    print(f"  HuBERT:")
    print(f"    - tsne_hubert.png")
    print(f"    - feature_coordinates_hubert.csv")
    print(f"    - features_hubert.npy")
    print("=" * 70)


# ============================================================================
# Main Execution
# ============================================================================
if __name__ == '__main__':
    # Configuration
    DATASET_DIR = './dataset'
    BATCH_SIZE = 32
    NUM_WORKERS = 0
    MAX_BATCHES = None  # Change to None to process all batches
    
    print("=" * 70)
    print("PRETRAINED FEATURE DISTRIBUTION ANALYSIS")
    print("=" * 70)
    
    # Check if dataset exists
    if not os.path.exists(DATASET_DIR):
        print(f"\nError: Dataset directory not found: {DATASET_DIR}")
        print("Please make sure the dataset directory exists.")
        exit(1)
    
    # Create dataloaders with CUDA optimization
    print(f"\nLoading dataset from: {DATASET_DIR}")
    train_loader, val_loader, dataset = create_dataloaders(
        dataset_dir=DATASET_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        use_cuda=torch.cuda.is_available()
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Run visualization
    visualize_pretrained_features(
        dataloader=train_loader,
        max_batches=MAX_BATCHES
    )
    
    print("\n" + "=" * 70)
    print("ALL DONE!")
    print("=" * 70)