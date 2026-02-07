"""
Supervised Contrastive Learning for Wav2Vec2 - Vietnamese Lexical Tones
- Finetune Wav2Vec2 to group same tones close together in representation space
- Visualize t-SNE after each epoch
- Save checkpoints and visualizations to Wav2vec_finetuned/
- Support weighted loss for class imbalance
"""

import os
os.environ["PYTHONWARNINGS"] = "ignore"
import warnings
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from tqdm import tqdm

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=UserWarning, module='urllib3')


# Import dataset from dataset.py
from dataset import LexicalSoundDataset, create_dataloaders

# ============================================================================
# Set Random Seeds
# ============================================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

# ============================================================================
# Configuration
# ============================================================================
# Dataset
DATASET_DIR = './dataset'

# Training
NUM_EPOCHS = 50
BATCH_SIZE = 256
TEST_BATCH_SIZE = 8  # For testing mode
MAX_BATCHES = None  # Set to None for full dataset, or number for testing (e.g., 8)
NUM_WORKERS = 4
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 1.0

# Weighted Loss Configuration
USE_WEIGHTED_LOSS = True  # Set to True to use weighted loss for class imbalance
WEIGHT_CALCULATION_METHOD = 'effective_samples'  # Options: 'inverse_freq', 'effective_samples', 'balanced'

# Contrastive Learning
TEMPERATURE = 0.07
PROJECTION_DIM = 128
CONTRASTIVE_WEIGHT = 0.5  # Balance between contrastive and classification loss

# Visualization
TSNE_SAMPLES = 10000  # Number of samples for t-SNE visualization

# Output
OUTPUT_DIR = 'Wav2vec_finetuned'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'checkpoints'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'visualizations'), exist_ok=True)

# Tone mapping and colors
TONE_NAMES = ['sắc', 'huyền', 'hỏi', 'ngã', 'nặng', 'không dấu']
NUM_CLASSES = len(TONE_NAMES)

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
# Calculate Class Weights
# ============================================================================
def calculate_class_weights(dataset, method='inverse_freq', beta=0.9999):
    """
    Calculate class weights for handling imbalanced datasets
    
    Args:
        dataset: Dataset object with labels
        method: Method to calculate weights
            - 'inverse_freq': 1 / frequency
            - 'effective_samples': (1 - beta) / (1 - beta^n)
            - 'balanced': sklearn-style balanced weights
        beta: Beta parameter for effective_samples method
    
    Returns:
        torch.Tensor: Class weights
    """
    # Count samples per class
    labels = []
    for i in range(len(dataset)):
        labels.append(dataset[i]['label'])
    
    labels = np.array(labels)
    class_counts = np.bincount(labels, minlength=NUM_CLASSES)
    
    # Calculate weights based on method
    if method == 'inverse_freq':
        # Inverse frequency: weight = 1 / frequency
        total_samples = len(labels)
        weights = total_samples / (NUM_CLASSES * class_counts)
        
    elif method == 'effective_samples':
        # Effective number of samples: https://arxiv.org/abs/1901.05555
        effective_num = 1.0 - np.power(beta, class_counts)
        weights = (1.0 - beta) / effective_num
        
    elif method == 'balanced':
        # Sklearn-style balanced weights
        total_samples = len(labels)
        weights = total_samples / (NUM_CLASSES * class_counts)
        
    else:
        raise ValueError(f"Unknown weight calculation method: {method}")
    
    # Normalize weights
    weights = weights / weights.sum() * NUM_CLASSES
    
    # Print weight information
    print("\n" + "=" * 80)
    print("CLASS WEIGHT INFORMATION")
    print("=" * 80)
    print(f"Method: {method}")
    if method == 'effective_samples':
        print(f"Beta: {beta}")
    print(f"\nClass distribution and weights:")
    print(f"{'Tone':<20} {'Count':<10} {'Percentage':<12} {'Weight':<10}")
    print("-" * 80)
    
    for i, tone_name in enumerate(TONE_NAMES):
        count = class_counts[i]
        percentage = count / len(labels) * 100
        weight = weights[i]
        english_name = TONE_ENGLISH_NAMES[tone_name]
        print(f"{english_name:<20} {count:<10} {percentage:>6.2f}%      {weight:>8.4f}")
    
    print("=" * 80)
    
    return torch.FloatTensor(weights)


# ============================================================================
# Supervised Contrastive Loss
# ============================================================================
class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss
    Same tone (same class) = positive pairs
    Different tones (different class) = negative pairs
    """
    def __init__(self, temperature=0.07):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        Args:
            features: [batch_size, projection_dim] - normalized embeddings
            labels: [batch_size] - class labels
        """
        batch_size = features.shape[0]

        # Normalize features
        features = F.normalize(features, dim=1)

        # Compute similarity matrix: [batch_size, batch_size]
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        # Create mask for positive pairs (same class)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)

        # Mask out diagonal (self-similarity)
        logits_mask = torch.ones_like(mask).scatter_(1,
            torch.arange(batch_size).view(-1, 1).to(features.device), 0)
        mask = mask * logits_mask

        # Compute log probabilities
        exp_logits = torch.exp(similarity_matrix) * logits_mask
        log_prob = similarity_matrix - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)

        # Compute mean of log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-9)

        # Loss is negative log-likelihood
        loss = -mean_log_prob_pos
        loss = loss.mean()

        return loss


# ============================================================================
# Wav2Vec2 Contrastive Model
# ============================================================================
class Wav2Vec2ContrastiveClassifier(nn.Module):
    """
    Wav2Vec2 model with Contrastive Learning
    - Encoder: Wav2Vec2 backbone
    - Projection head: For contrastive learning
    - Classification head: For supervised classification
    """

    def __init__(self, model_name='facebook/wav2vec2-base', num_classes=6,
                 projection_dim=128, freeze_feature_extractor=False):
        super(Wav2Vec2ContrastiveClassifier, self).__init__()

        # Load Wav2Vec2 model
        print(f"\nLoading Wav2Vec2 model: {model_name}")
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)

        if freeze_feature_extractor:
            for param in self.wav2vec2.feature_extractor.parameters():
                param.requires_grad = False
            print("Feature extractor frozen")
        else:
            print("Feature extractor trainable")

        hidden_size = self.wav2vec2.config.hidden_size

        # Encoder projection (shared representation)
        self.encoder = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, projection_dim)
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

        print("Wav2Vec2 Contrastive model initialized")

    def forward(self, input_values, return_all=False):
        # Extract Wav2Vec2 features
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs.last_hidden_state

        # Average pooling
        pooled = hidden_states.mean(dim=1)  # [batch_size, hidden_dim]

        # Shared encoder
        encoded = self.encoder(pooled)  # [batch_size, 512]

        # Projection for contrastive learning
        projected = self.projection_head(encoded)  # [batch_size, projection_dim]

        # Classification logits
        logits = self.classifier(encoded)  # [batch_size, num_classes]

        if return_all:
            return logits, projected, encoded
        return logits, projected


# ============================================================================
# Training Functions
# ============================================================================
def get_lr(optimizer):
    """Get current learning rate"""
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train_epoch(model, dataloader, ce_criterion, contrastive_criterion,
                optimizer, device, epoch, contrastive_weight=0.5, max_batches=None):
    """
    Train for one epoch with combined loss
    Loss = (1 - α) * CrossEntropy + α * ContrastiveLoss
    """
    model.train()
    total_loss = 0
    total_ce_loss = 0
    total_cont_loss = 0
    all_preds = []
    all_labels = []

    # Limit batches if max_batches is set
    total_batches = len(dataloader) if max_batches is None else min(max_batches, len(dataloader))
    
    pbar = tqdm(enumerate(dataloader), total=total_batches, desc=f"Epoch {epoch}/{NUM_EPOCHS}")
    
    for batch_idx, batch in pbar:
        if max_batches is not None and batch_idx >= max_batches:
            break
            
        waveforms = batch['waveform'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        # Forward pass
        logits, projected = model(waveforms)

        # Cross-Entropy Loss (with optional weights)
        ce_loss = ce_criterion(logits, labels)

        # Contrastive Loss
        cont_loss = contrastive_criterion(projected, labels)

        # Combined loss
        loss = (1 - contrastive_weight) * ce_loss + contrastive_weight * cont_loss

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        optimizer.step()

        # Metrics
        total_loss += loss.item()
        total_ce_loss += ce_loss.item()
        total_cont_loss += cont_loss.item()

        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        # Update progress bar
        current_lr = get_lr(optimizer)
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'ce': f'{ce_loss.item():.4f}',
            'cont': f'{cont_loss.item():.4f}',
            'lr': f'{current_lr:.2e}'
        })

    batches_processed = min(batch_idx + 1, total_batches)
    avg_loss = total_loss / batches_processed
    avg_ce_loss = total_ce_loss / batches_processed
    avg_cont_loss = total_cont_loss / batches_processed
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, avg_ce_loss, avg_cont_loss, accuracy


def validate(model, dataloader, ce_criterion, contrastive_criterion, device,
             contrastive_weight=0.5, max_batches=None):
    """Validate model"""
    model.eval()
    total_loss = 0
    total_ce_loss = 0
    total_cont_loss = 0
    all_preds = []
    all_labels = []

    # Limit batches if max_batches is set
    total_batches = len(dataloader) if max_batches is None else min(max_batches, len(dataloader))

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, total=total_batches, desc="Validation", leave=False)):
            if max_batches is not None and batch_idx >= max_batches:
                break
                
            waveforms = batch['waveform'].to(device)
            labels = batch['label'].to(device)

            logits, projected = model(waveforms)

            ce_loss = ce_criterion(logits, labels)
            cont_loss = contrastive_criterion(projected, labels)
            loss = (1 - contrastive_weight) * ce_loss + contrastive_weight * cont_loss

            total_loss += loss.item()
            total_ce_loss += ce_loss.item()
            total_cont_loss += cont_loss.item()

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    batches_processed = min(batch_idx + 1, total_batches)
    avg_loss = total_loss / batches_processed
    avg_ce_loss = total_ce_loss / batches_processed
    avg_cont_loss = total_cont_loss / batches_processed
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, avg_ce_loss, avg_cont_loss, accuracy, all_preds, all_labels


# ============================================================================
# Visualization Functions
# ============================================================================
def extract_features_for_tsne(model, dataloader, sample_size=500, device='cpu', max_batches=None):
    """Extract features for t-SNE visualization"""
    model.eval()
    all_features = []
    all_labels = []
    all_tone_names = []

    samples_collected = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            if samples_collected >= sample_size:
                break

            waveforms = batch['waveform'].to(device)
            labels = batch['label'].cpu().numpy()
            tone_names = batch['tone_name']

            # Extract encoded features (before projection/classification)
            _, _, features = model(waveforms, return_all=True)
            all_features.append(features.cpu().numpy())
            all_labels.extend(labels)
            all_tone_names.extend(tone_names)

            samples_collected += len(labels)

    features = np.vstack(all_features)

    # Limit to sample_size
    if len(all_labels) > sample_size:
        indices = np.random.choice(len(all_labels), sample_size, replace=False)
        features = features[indices]
        all_labels = [all_labels[i] for i in indices]
        all_tone_names = [all_tone_names[i] for i in indices]

    return features, all_labels, all_tone_names


def extract_all_features(model, dataloader, device='cpu', max_batches=None, desc="Extracting features"):
    """Extract features from entire dataloader without sampling"""
    model.eval()
    all_features = []
    all_labels = []
    all_tone_names = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=desc)):
            if max_batches is not None and batch_idx >= max_batches:
                break

            waveforms = batch['waveform'].to(device)
            labels = batch['label'].cpu().numpy()
            tone_names = batch['tone_name']

            # Extract encoded features (before projection/classification)
            _, _, features = model(waveforms, return_all=True)
            all_features.append(features.cpu().numpy())
            all_labels.extend(labels)
            all_tone_names.extend(tone_names)

    features = np.vstack(all_features)
    return features, all_labels, all_tone_names


def plot_tsne(features, labels, epoch, save_path, title_suffix=""):
    """Plot t-SNE visualization"""
    print(f"\n  Running t-SNE on {len(labels)} samples...")

    perplexity = min(30, (len(labels) - 1) // 3)
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=1000,
                random_state=42, verbose=0)
    features_2d = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=(14, 10))

    for tone_idx, tone_name in enumerate(TONE_NAMES):
        mask = np.array(labels) == tone_idx
        if mask.sum() > 0:
            color = TONE_COLORS[tone_name]
            english_name = TONE_ENGLISH_NAMES[tone_name]

            ax.scatter(
                features_2d[mask, 0],
                features_2d[mask, 1],
                c=color,
                label=english_name,
                alpha=0.6,
                s=40,
                edgecolors='white',
                linewidth=0.5
            )

    title = f'Vietnamese Tone Distribution - t-SNE Visualization (Wav2Vec2)\nEpoch {epoch} {title_suffix}'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('t-SNE Dimension 1', fontsize=18)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=18)
    ax.legend(title='Tone', fontsize=16, title_fontsize=16,
              loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  t-SNE saved to: {save_path}")


def plot_tsne_with_splits(features, labels, splits, save_path, title="Full Dataset t-SNE"):
    """
    Plot t-SNE visualization with train/val split indicators
    
    Args:
        features: Feature vectors
        labels: Class labels
        splits: List of 'train' or 'val' for each sample
        save_path: Path to save the plot
        title: Title for the plot
    """
    print(f"\n  Running t-SNE on {len(labels)} samples (full dataset)...")
    
    perplexity = min(30, (len(labels) - 1) // 3)
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=1000,
                random_state=42, verbose=0)
    features_2d = tsne.fit_transform(features)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(28, 10))
    
    # Left plot: Colored by tone
    for tone_idx, tone_name in enumerate(TONE_NAMES):
        mask = np.array(labels) == tone_idx
        if mask.sum() > 0:
            color = TONE_COLORS[tone_name]
            english_name = TONE_ENGLISH_NAMES[tone_name]
            
            ax1.scatter(
                features_2d[mask, 0],
                features_2d[mask, 1],
                c=color,
                label=english_name,
                alpha=0.6,
                s=40,
                edgecolors='white',
                linewidth=0.5
            )
    
    ax1.set_title(f'{title} (Wav2Vec2)\nColored by Tone', fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlabel('t-SNE Dimension 1', fontsize=14)
    ax1.set_ylabel('t-SNE Dimension 2', fontsize=14)
    ax1.legend(title='Tone', fontsize=12, title_fontsize=12, loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Right plot: Colored by train/val split
    splits_array = np.array(splits)
    train_mask = splits_array == 'train'
    val_mask = splits_array == 'val'
    
    if train_mask.sum() > 0:
        ax2.scatter(
            features_2d[train_mask, 0],
            features_2d[train_mask, 1],
            c='#3498db',
            label=f'Train ({train_mask.sum()} samples)',
            alpha=0.5,
            s=30,
            edgecolors='white',
            linewidth=0.5
        )
    
    if val_mask.sum() > 0:
        ax2.scatter(
            features_2d[val_mask, 0],
            features_2d[val_mask, 1],
            c='#e74c3c',
            label=f'Validation ({val_mask.sum()} samples)',
            alpha=0.5,
            s=30,
            edgecolors='white',
            linewidth=0.5
        )
    
    ax2.set_title(f'{title} (Wav2Vec2)\nColored by Dataset Split', fontsize=16, fontweight='bold', pad=20)
    ax2.set_xlabel('t-SNE Dimension 1', fontsize=14)
    ax2.set_ylabel('t-SNE Dimension 2', fontsize=14)
    ax2.legend(fontsize=12, loc='best', framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Full dataset t-SNE saved to: {save_path}")


def visualize_full_dataset_tsne(model, train_loader, val_loader, device, output_dir, max_batches=None):
    """
    Visualize t-SNE on full dataset (train + val) after training
    
    Args:
        model: Trained model
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to run on
        output_dir: Directory to save visualizations
        max_batches: Maximum batches to process (None for full dataset)
    """
    print("\n" + "=" * 80)
    print("STEP 6: VISUALIZING FULL DATASET WITH t-SNE")
    print("=" * 80)
    
    model.eval()
    
    # Extract features from training set
    print("\nExtracting features from training set...")
    train_features, train_labels, train_tone_names = extract_all_features(
        model, train_loader, device, max_batches, desc="Processing train set"
    )
    
    # Extract features from validation set
    print("\nExtracting features from validation set...")
    val_features, val_labels, val_tone_names = extract_all_features(
        model, val_loader, device, max_batches, desc="Processing validation set"
    )
    
    # Combine train and val
    print("\nCombining train and validation features...")
    all_features = np.vstack([train_features, val_features])
    all_labels = train_labels + val_labels
    all_splits = ['train'] * len(train_labels) + ['val'] * len(val_labels)
    
    total_samples = len(all_labels)
    print(f"  Total samples: {total_samples}")
    print(f"    Train: {len(train_labels)} ({len(train_labels)/total_samples*100:.1f}%)")
    print(f"    Val: {len(val_labels)} ({len(val_labels)/total_samples*100:.1f}%)")
    
    # Sample if dataset is too large (for faster t-SNE)
    max_tsne_samples = 15000
    if total_samples > max_tsne_samples:
        print(f"\n  Dataset is large ({total_samples} samples), sampling {max_tsne_samples} for t-SNE...")
        indices = np.random.choice(total_samples, max_tsne_samples, replace=False)
        all_features = all_features[indices]
        all_labels = [all_labels[i] for i in indices]
        all_splits = [all_splits[i] for i in indices]
        print(f"  Sampled {len(all_labels)} samples for visualization")
    
    # Create visualizations
    print("\nGenerating t-SNE visualizations...")
    
    # 1. Full dataset with split visualization
    plot_tsne_with_splits(
        all_features, 
        all_labels, 
        all_splits,
        os.path.join(output_dir, 'visualizations', 'tsne_full_dataset.png'),
        title="Full Dataset - Final Model"
    )
    
    # 2. Train-only visualization
    train_mask = np.array(all_splits) == 'train'
    if train_mask.sum() > 0:
        plot_tsne(
            all_features[train_mask],
            [all_labels[i] for i, m in enumerate(train_mask) if m],
            "Final",
            os.path.join(output_dir, 'visualizations', 'tsne_train_only.png'),
            title_suffix="(Train Set Only)"
        )
    
    # 3. Val-only visualization
    val_mask = np.array(all_splits) == 'val'
    if val_mask.sum() > 0:
        plot_tsne(
            all_features[val_mask],
            [all_labels[i] for i, m in enumerate(val_mask) if m],
            "Final",
            os.path.join(output_dir, 'visualizations', 'tsne_val_only.png'),
            title_suffix="(Validation Set Only)"
        )
    
    # Generate statistics per tone
    print("\n" + "=" * 80)
    print("Dataset Statistics by Tone:")
    print("=" * 80)
    
    tone_stats = []
    for tone_idx, tone_name in enumerate(TONE_NAMES):
        tone_mask = np.array(all_labels) == tone_idx
        total_tone = tone_mask.sum()
        
        if total_tone > 0:
            train_tone = np.sum([1 for i, (label, split) in enumerate(zip(all_labels, all_splits)) 
                                if label == tone_idx and split == 'train'])
            val_tone = total_tone - train_tone
            
            english_name = TONE_ENGLISH_NAMES[tone_name]
            print(f"\n{english_name} ({tone_name}):")
            print(f"  Total: {total_tone} samples ({total_tone/len(all_labels)*100:.1f}%)")
            print(f"  Train: {train_tone} samples")
            print(f"  Val: {val_tone} samples")
            
            tone_stats.append({
                'Tone': tone_name,
                'English': english_name,
                'Total': total_tone,
                'Train': train_tone,
                'Val': val_tone,
                'Percentage': f"{total_tone/len(all_labels)*100:.1f}%"
            })
    
    # Save statistics to CSV
    df_stats = pd.DataFrame(tone_stats)
    stats_path = os.path.join(output_dir, 'dataset_statistics.csv')
    df_stats.to_csv(stats_path, index=False)
    print(f"\nDataset statistics saved to: {stats_path}")
    
    print("\n" + "=" * 80)
    print("Full dataset visualization completed!")
    print("=" * 80)
    print(f"\nGenerated visualizations:")
    print(f"  {os.path.join(output_dir, 'visualizations', 'tsne_full_dataset.png')}")
    print(f"  {os.path.join(output_dir, 'visualizations', 'tsne_train_only.png')}")
    print(f"  {os.path.join(output_dir, 'visualizations', 'tsne_val_only.png')}")
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================================
# Main Training Function
# ============================================================================
def main():
    # Device detection (done here to avoid multiple prints from worker processes)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"CUDA is available!")
        print(f"   Device: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        device = torch.device('cpu')
        print(f"CUDA not available, using CPU")

    print(f"Using device: {device}\n")
    
    print("=" * 80)
    print("SUPERVISED CONTRASTIVE LEARNING FOR WAV2VEC2")
    print("Vietnamese Lexical Tone Classification")
    if USE_WEIGHTED_LOSS:
        print(f"WITH WEIGHTED LOSS ({WEIGHT_CALCULATION_METHOD})")
    print("=" * 80)

    # Determine batch size based on mode
    if MAX_BATCHES is None:
        actual_batch_size = BATCH_SIZE
        print(f"\nFULL DATASET MODE")
        print(f"   Batch size: {actual_batch_size}")
    else:
        actual_batch_size = TEST_BATCH_SIZE
        print(f"\nTEST MODE")
        print(f"   Max batches: {MAX_BATCHES}")
        print(f"   Batch size: {actual_batch_size}")

    # ========================================================================
    # Sanity Check Dataset
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 1: SANITY CHECK DATASET")
    print("=" * 80)

    if not os.path.exists(DATASET_DIR):
        print(f"\nERROR: Dataset directory not found: {DATASET_DIR}")
        print("Please make sure the dataset directory exists.")
        return

    print(f"\nDataset directory: {DATASET_DIR}")
    print("Scanning and validating audio files...")

    # Create dataset (will perform sanity check during initialization)
    train_loader, val_loader, dataset = create_dataloaders(
        dataset_dir=DATASET_DIR,
        batch_size=actual_batch_size,
        num_workers=NUM_WORKERS,
        use_cuda=torch.cuda.is_available()
    )

    print(f"\nDataset validated successfully!")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")

    # ========================================================================
    # Calculate Class Weights (if enabled)
    # ========================================================================
    class_weights = None
    if USE_WEIGHTED_LOSS:
        print("\n" + "=" * 80)
        print("CALCULATING CLASS WEIGHTS")
        print("=" * 80)
        class_weights = calculate_class_weights(
            dataset, 
            method=WEIGHT_CALCULATION_METHOD
        )
        class_weights = class_weights.to(device)

    # ========================================================================
    # Initialize Model
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: INITIALIZE MODEL")
    print("=" * 80)

    model = Wav2Vec2ContrastiveClassifier(
        model_name='facebook/wav2vec2-base',
        num_classes=NUM_CLASSES,
        projection_dim=PROJECTION_DIM,
        freeze_feature_extractor=False
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Projection dimension: {PROJECTION_DIM}")
    print(f"  Temperature: {TEMPERATURE}")
    print(f"  Contrastive weight: {CONTRASTIVE_WEIGHT}")
    if USE_WEIGHTED_LOSS:
        print(f"  Using weighted loss: YES ({WEIGHT_CALCULATION_METHOD})")
    else:
        print(f"  Using weighted loss: NO")

    # ========================================================================
    # Setup Training
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: SETUP TRAINING")
    print("=" * 80)

    # Loss functions
    ce_criterion = nn.CrossEntropyLoss(weight=class_weights)
    contrastive_criterion = SupervisedContrastiveLoss(temperature=TEMPERATURE)

    # Optimizer with different learning rates
    optimizer = optim.AdamW([
        {'params': model.wav2vec2.parameters(), 'lr': LEARNING_RATE * 0.1},
        {'params': model.encoder.parameters(), 'lr': LEARNING_RATE},
        {'params': model.projection_head.parameters(), 'lr': LEARNING_RATE},
        {'params': model.classifier.parameters(), 'lr': LEARNING_RATE}
    ], weight_decay=WEIGHT_DECAY)

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
    )

    # Training history
    history = {
        'train_loss': [],
        'train_ce_loss': [],
        'train_cont_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_ce_loss': [],
        'val_cont_loss': [],
        'val_acc': [],
        'learning_rate': []
    }

    best_val_acc = 0.0

    print(f"\nTraining Configuration:")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Batch size: {actual_batch_size}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Weight decay: {WEIGHT_DECAY}")
    print(f"  Gradient clip: {GRADIENT_CLIP}")

    # ========================================================================
    # Training Loop
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 4: TRAINING")
    print("=" * 80)

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{NUM_EPOCHS}")
        print(f"{'='*80}")

        # Train
        train_loss, train_ce, train_cont, train_acc = train_epoch(
            model, train_loader, ce_criterion, contrastive_criterion,
            optimizer, device, epoch, CONTRASTIVE_WEIGHT, MAX_BATCHES
        )

        # Validate
        val_loss, val_ce, val_cont, val_acc, val_preds, val_labels = validate(
            model, val_loader, ce_criterion, contrastive_criterion,
            device, CONTRASTIVE_WEIGHT, MAX_BATCHES
        )

        # Update scheduler
        scheduler.step()
        current_lr = get_lr(optimizer)

        # Save history
        history['train_loss'].append(train_loss)
        history['train_ce_loss'].append(train_ce)
        history['train_cont_loss'].append(train_cont)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_ce_loss'].append(val_ce)
        history['val_cont_loss'].append(val_cont)
        history['val_acc'].append(val_acc)
        history['learning_rate'].append(current_lr)

        # Print metrics
        print(f"\nEpoch {epoch} Results:")
        print(f"  Train Loss: {train_loss:.4f} (CE: {train_ce:.4f}, Cont: {train_cont:.4f}) | Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} (CE: {val_ce:.4f}, Cont: {val_cont:.4f}) | Acc: {val_acc:.4f}")
        print(f"  Learning Rate: {current_lr:.2e}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            
            # Save checkpoint ready for HuggingFace
            checkpoint_path = os.path.join(OUTPUT_DIR, 'checkpoints', 'best_model')
            os.makedirs(checkpoint_path, exist_ok=True)
            
            # Save model state
            model.wav2vec2.save_pretrained(checkpoint_path)
            
            # Save additional components
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': model.encoder.state_dict(),
                'projection_head_state_dict': model.projection_head.state_dict(),
                'classifier_state_dict': model.classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'class_weights': class_weights.cpu() if class_weights is not None else None,
                'config': {
                    'num_classes': NUM_CLASSES,
                    'projection_dim': PROJECTION_DIM,
                    'temperature': TEMPERATURE,
                    'use_weighted_loss': USE_WEIGHTED_LOSS,
                    'weight_calculation_method': WEIGHT_CALCULATION_METHOD if USE_WEIGHTED_LOSS else None
                }
            }, os.path.join(checkpoint_path, 'additional_components.pt'))
            
            print(f"  Best model saved! (Val Acc: {val_acc:.4f})")

        # Visualize t-SNE every epoch
        print(f"\n  Generating t-SNE visualization for epoch {epoch}...")
        features, labels, tones = extract_features_for_tsne(
            model, val_loader, sample_size=TSNE_SAMPLES, device=device, max_batches=MAX_BATCHES
        )
        plot_tsne(
            features, labels, epoch,
            os.path.join(OUTPUT_DIR, 'visualizations', f'tsne_epoch_{epoch:02d}.png'),
            title_suffix=f"(Val Acc: {val_acc:.4f})"
        )

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ========================================================================
    # Final Evaluation
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 5: FINAL EVALUATION")
    print("=" * 80)

    # Load best model
    print("\nLoading best model...")
    checkpoint_path = os.path.join(OUTPUT_DIR, 'checkpoints', 'best_model')
    checkpoint_data = torch.load(os.path.join(checkpoint_path, 'additional_components.pt'))
    
    model.encoder.load_state_dict(checkpoint_data['encoder_state_dict'])
    model.projection_head.load_state_dict(checkpoint_data['projection_head_state_dict'])
    model.classifier.load_state_dict(checkpoint_data['classifier_state_dict'])
    
    print(f"Best model loaded (Epoch {checkpoint_data['epoch']}, Val Acc: {checkpoint_data['val_acc']:.4f})")

    # Final validation
    val_loss, val_ce, val_cont, val_acc, val_preds, val_labels = validate(
        model, val_loader, ce_criterion, contrastive_criterion, device, CONTRASTIVE_WEIGHT, MAX_BATCHES
    )

    # Classification report
    print("\n" + "=" * 80)
    print("Classification Report:")
    print("=" * 80)
    unique_labels = sorted(set(val_labels))
    target_names = [TONE_NAMES[i] for i in unique_labels]
    print(classification_report(val_labels, val_preds, labels=unique_labels,
                                target_names=target_names, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(val_labels, val_preds, labels=unique_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    
    title = 'Confusion Matrix - Supervised Contrastive Learning (Wav2Vec2)'
    if USE_WEIGHTED_LOSS:
        title += f'\nWeighted Loss ({WEIGHT_CALCULATION_METHOD})'
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nConfusion matrix saved")

    # Plot training curves
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    epochs_range = range(1, len(history['train_loss']) + 1)

    # Total Loss
    axes[0, 0].plot(epochs_range, history['train_loss'], 'b-', label='Train', linewidth=2)
    axes[0, 0].plot(epochs_range, history['val_loss'], 'r-', label='Val', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Total Loss', fontsize=12)
    title = 'Total Loss (CE + Contrastive)'
    if USE_WEIGHTED_LOSS:
        title += f'\nWeighted CE ({WEIGHT_CALCULATION_METHOD})'
    axes[0, 0].set_title(title, fontweight='bold', fontsize=14)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # CE Loss
    axes[0, 1].plot(epochs_range, history['train_ce_loss'], 'b-', label='Train CE', linewidth=2)
    axes[0, 1].plot(epochs_range, history['val_ce_loss'], 'r-', label='Val CE', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Cross-Entropy Loss', fontsize=12)
    title = 'Cross-Entropy Loss'
    if USE_WEIGHTED_LOSS:
        title += f' (Weighted - {WEIGHT_CALCULATION_METHOD})'
    axes[0, 1].set_title(title, fontweight='bold', fontsize=14)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Contrastive Loss
    axes[1, 0].plot(epochs_range, history['train_cont_loss'], 'b-', label='Train Cont', linewidth=2)
    axes[1, 0].plot(epochs_range, history['val_cont_loss'], 'r-', label='Val Cont', linewidth=2)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Contrastive Loss', fontsize=12)
    axes[1, 0].set_title('Contrastive Loss', fontweight='bold', fontsize=14)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Accuracy
    axes[1, 1].plot(epochs_range, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[1, 1].plot(epochs_range, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Accuracy', fontsize=12)
    axes[1, 1].set_title('Accuracy', fontweight='bold', fontsize=14)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved")

    # Save history
    df_history = pd.DataFrame(history)
    df_history.to_csv(os.path.join(OUTPUT_DIR, 'training_history.csv'), index=False)
    print(f"Training history saved")

    # ========================================================================
    # Visualize Full Dataset
    # ========================================================================
    visualize_full_dataset_tsne(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        output_dir=OUTPUT_DIR,
        max_batches=MAX_BATCHES
    )

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("TRAINING COMPLETED!")
    print("=" * 80)
    print(f"\nBest Validation Accuracy: {best_val_acc:.4f}")
    print(f"Final Validation Accuracy: {val_acc:.4f}")
    print(f"Total epochs trained: {len(history['train_loss'])}")
    if USE_WEIGHTED_LOSS:
        print(f"Weighted Loss Method: {WEIGHT_CALCULATION_METHOD}")
    print(f"\nOutput directory: {OUTPUT_DIR}/")
    print(f"\nGenerated files:")
    print(f"  checkpoints/")
    print(f"     best_model/ (Ready for HuggingFace)")
    print(f"         config.json")
    print(f"         pytorch_model.bin")
    print(f"         additional_components.pt")
    print(f"  visualizations/")
    print(f"     tsne_epoch_XX.png ({NUM_EPOCHS} files)")
    print(f"     tsne_full_dataset.png")
    print(f"     tsne_train_only.png")
    print(f"     tsne_val_only.png")
    print(f"  confusion_matrix.png")
    print(f"  training_curves.png")
    print(f"  training_history.csv")
    print(f"  dataset_statistics.csv")
    print("=" * 80)
    print("\nAll done!")


if __name__ == '__main__':
    main()