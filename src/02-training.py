"""
Training Script for Legal Text Readability Classification

This script implements:
- Baseline model (TF-IDF + Logistic Regression)
- Fine-tuned Hungarian BERT model
- Mixed precision training
- Early stopping and checkpointing
- Comprehensive logging
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
import gc
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
import warnings
warnings.filterwarnings('ignore')

# Import configuration
import config

# ============================================================================
# SETUP LOGGING
# ============================================================================

def setup_logger():
    """Setup logging to both file and console."""
    log_file = Path(config.LOG_FILE)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('training')
    logger.setLevel(logging.INFO)
    
    # File handler
    fh = logging.FileHandler(log_file, mode='w')
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

logger = setup_logger()

# ============================================================================
# REPRODUCIBILITY
# ============================================================================

def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to: {seed}")


# ============================================================================
# DATA LOADING
# ============================================================================

class TextDataset(Dataset):
    """
    Custom Dataset for text classification with BERT tokenization.
    """
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label - 1, dtype=torch.long)  # 0-indexed
        }


def load_data():
    """Load train and validation datasets."""
    logger.info("\n" + "="*60)
    logger.info("DATA LOADING")
    logger.info("="*60)
    
    train_df = pd.read_csv(config.DATA_DIR / "train.csv")
    val_df = pd.read_csv(config.DATA_DIR / "val.csv")
    
    logger.info(f"✓ Train samples: {len(train_df)}")
    logger.info(f"✓ Validation samples: {len(val_df)}")
    
    # Log label distribution
    logger.info("\nTrain label distribution:")
    for label, count in train_df['label'].value_counts().sort_index().items():
        pct = (count / len(train_df)) * 100
        logger.info(f"  Label {label}: {count:4d} ({pct:5.1f}%)")
    
    return train_df, val_df


def create_dataloaders(train_df, val_df, tokenizer):
    """Create PyTorch DataLoaders."""
    train_dataset = TextDataset(
        train_df['text'].values,
        train_df['label'].values,
        tokenizer,
        config.MAX_LENGTH
    )
    
    val_dataset = TextDataset(
        val_df['text'].values,
        val_df['label'].values,
        tokenizer,
        config.MAX_LENGTH
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Reduced to save memory
        pin_memory=False  # Disabled to save memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,  # Reduced to save memory
        pin_memory=False  # Disabled to save memory
    )
    
    logger.info(f"\n✓ DataLoaders created")
    logger.info(f"  Batch size: {config.BATCH_SIZE}")
    logger.info(f"  Train batches: {len(train_loader)}")
    logger.info(f"  Val batches: {len(val_loader)}")
    
    return train_loader, val_loader


# ============================================================================
# BASELINE MODEL
# ============================================================================

def train_baseline(train_df, val_df):
    """
    Train baseline model: TF-IDF + Logistic Regression.
    Provides a simple benchmark for comparison.
    """
    logger.info("\n" + "="*60)
    logger.info("BASELINE MODEL")
    logger.info("="*60)
    logger.info("Model: TF-IDF + Logistic Regression")
    
    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(
        max_features=config.BASELINE_MAX_FEATURES,
        ngram_range=config.BASELINE_NGRAM_RANGE,
        min_df=2
    )
    
    X_train = vectorizer.fit_transform(train_df['text'])
    X_val = vectorizer.transform(val_df['text'])
    y_train = train_df['label'].values
    y_val = val_df['label'].values
    
    # Train logistic regression
    clf = LogisticRegression(
        multi_class='multinomial',
        max_iter=1000,
        random_state=config.RANDOM_SEED
    )
    clf.fit(X_train, y_train)
    
    # Evaluate
    train_pred = clf.predict(X_train)
    val_pred = clf.predict(X_val)
    
    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)
    val_f1 = f1_score(y_val, val_pred, average='weighted')
    
    logger.info(f"\nBaseline Results:")
    logger.info(f"  Train Accuracy: {train_acc:.4f}")
    logger.info(f"  Val Accuracy: {val_acc:.4f}")
    logger.info(f"  Val F1-Score: {val_f1:.4f}")
    
    return {
        'train_acc': train_acc,
        'val_acc': val_acc,
        'val_f1': val_f1
    }


# ============================================================================
# BERT MODEL
# ============================================================================

class HungarianBERTClassifier(nn.Module):
    """
    Hungarian BERT model with classification head.
    
    Architecture:
    - Pre-trained BERT encoder
    - Dropout for regularization
    - Two-layer classification head
    """
    def __init__(self, num_classes=config.NUM_CLASSES, dropout=config.DROPOUT_RATE):
        super(HungarianBERTClassifier, self).__init__()
        
        # Load pre-trained BERT
        self.bert = AutoModel.from_pretrained(config.MODEL_NAME)
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(config.HIDDEN_SIZE, config.CLASSIFIER_HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(config.CLASSIFIER_HIDDEN_SIZE, num_classes)
        )
    
    def forward(self, input_ids, attention_mask):
        # BERT encoding
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        
        # Classification
        logits = self.classifier(pooled_output)
        
        return logits


def count_parameters(model):
    """Count trainable parameters in model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ============================================================================
# EARLY STOPPING
# ============================================================================

class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    Stops training if validation loss doesn't improve for patience epochs.
    """
    def __init__(self, patience=config.EARLY_STOPPING_PATIENCE, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_epoch = 0
    
    def __call__(self, val_loss, epoch):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_epoch = epoch
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info(f"\nEarly stopping triggered after {self.counter} epochs without improvement")
        else:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
        
        return self.early_stop


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_epoch(model, dataloader, optimizer, criterion, scheduler, scaler, device, epoch):
    """Train for one epoch with gradient accumulation."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        # Mixed precision forward pass
        with autocast(enabled=config.USE_MIXED_PRECISION):
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            # Scale loss for gradient accumulation
            loss = loss / config.ACCUMULATION_STEPS
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        
        # Optimizer step every ACCUMULATION_STEPS
        if (batch_idx + 1) % config.ACCUMULATION_STEPS == 0:
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP_VALUE)
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
        
        # Metrics (use unscaled loss for logging)
        total_loss += loss.item() * config.ACCUMULATION_STEPS
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Log progress
        if (batch_idx + 1) % config.LOG_INTERVAL == 0:
            logger.info(f"  Batch [{batch_idx+1}/{len(dataloader)}] "
                       f"Loss: {loss.item()*config.ACCUMULATION_STEPS:.4f} "
                       f"Acc: {100*correct/total:.2f}%")
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return avg_loss, accuracy, f1, all_preds, all_labels


# ============================================================================
# CHECKPOINTING
# ============================================================================

def save_checkpoint(model, optimizer, epoch, val_loss, val_acc, val_f1, is_best=False):
    """Save model checkpoint."""
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_acc': val_acc,
        'val_f1': val_f1
    }
    
    if is_best:
        path = config.CHECKPOINT_DIR / "best_model.pt"
        torch.save(checkpoint, path)
        logger.info(f"  ✓ Best model saved (Val Loss: {val_loss:.4f})")


def load_checkpoint(model, optimizer, checkpoint_path):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['val_loss']


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def main():
    """Main training pipeline."""
    
    # Header
    logger.info("\n" + "="*60)
    logger.info("LEGAL TEXT READABILITY CLASSIFICATION")
    logger.info("="*60)
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Log hyperparameters
    logger.info("\n" + "="*60)
    logger.info("HYPERPARAMETERS")
    logger.info("="*60)
    logger.info(f"Model: {config.MODEL_NAME}")
    logger.info(f"Epochs: {config.EPOCHS}")
    logger.info(f"Batch Size: {config.BATCH_SIZE}")
    logger.info(f"Learning Rate: {config.LEARNING_RATE}")
    logger.info(f"Weight Decay: {config.WEIGHT_DECAY}")
    logger.info(f"Dropout Rate: {config.DROPOUT_RATE}")
    logger.info(f"Max Length: {config.MAX_LENGTH}")
    logger.info(f"Warmup Ratio: {config.WARMUP_RATIO}")
    logger.info(f"Early Stopping Patience: {config.EARLY_STOPPING_PATIENCE}")
    logger.info(f"Gradient Clip: {config.GRADIENT_CLIP_VALUE}")
    logger.info(f"Label Smoothing: {config.LABEL_SMOOTHING}")
    logger.info(f"Mixed Precision: {config.USE_MIXED_PRECISION}")
    logger.info(f"Random Seed: {config.RANDOM_SEED}")
    
    # Set seed
    set_seed(config.RANDOM_SEED)
    
    # Check GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"\n✓ Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Load data
    train_df, val_df = load_data()
    
    # Train baseline
    baseline_results = train_baseline(train_df, val_df)
    
    # Clear memory after baseline
    gc.collect()
    
    # Initialize tokenizer
    logger.info("\n" + "="*60)
    logger.info("INITIALIZING BERT MODEL")
    logger.info("="*60)
    logger.info(f"Loading tokenizer: {config.MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(train_df, val_df, tokenizer)
    
    # Initialize model
    logger.info(f"\nLoading model: {config.MODEL_NAME}")
    model = HungarianBERTClassifier().to(device)
    
    total_params, trainable_params = count_parameters(model)
    logger.info(f"\nModel Architecture:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    
    # Setup training
    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    total_steps = len(train_loader) * config.EPOCHS
    warmup_steps = int(total_steps * config.WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    logger.info(f"\nTraining Configuration:")
    logger.info(f"  Total steps: {total_steps}")
    logger.info(f"  Warmup steps: {warmup_steps}")
    
    # Mixed precision scaler
    scaler = GradScaler(enabled=config.USE_MIXED_PRECISION)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE)
    
    # Training loop
    logger.info("\n" + "="*60)
    logger.info("TRAINING")
    logger.info("="*60)
    
    best_val_loss = float('inf')
    
    for epoch in range(config.EPOCHS):
        logger.info(f"\nEpoch {epoch+1}/{config.EPOCHS}")
        logger.info("-" * 60)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, scheduler, scaler, device, epoch
        )
        
        # Validate
        val_loss, val_acc, val_f1, val_preds, val_labels = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # Log metrics
        logger.info(f"\nEpoch {epoch+1} Results:")
        logger.info(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        logger.info(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        
        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss, val_acc, val_f1, is_best=True)
        
        # Early stopping
        if early_stopping(val_loss, epoch):
            logger.info(f"\nBest epoch: {early_stopping.best_epoch + 1}")
            break
    
    # Load best model
    logger.info("\n" + "="*60)
    logger.info("FINAL RESULTS")
    logger.info("="*60)
    
    checkpoint_path = config.CHECKPOINT_DIR / "best_model.pt"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"\n✓ Loaded best model from epoch {checkpoint['epoch']+1}")
        logger.info(f"  Best Val Loss: {checkpoint['val_loss']:.4f}")
        logger.info(f"  Best Val Acc: {checkpoint['val_acc']:.4f}")
        logger.info(f"  Best Val F1: {checkpoint['val_f1']:.4f}")
        
        # Final validation
        val_loss, val_acc, val_f1, val_preds, val_labels = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # Classification report
        logger.info("\nClassification Report:")
        # Convert 0-indexed back to 1-indexed for display
        report = classification_report(
            [l+1 for l in val_labels],
            [p+1 for p in val_preds],
            target_names=[config.LABEL_NAMES[i] for i in range(1, 6)],
            digits=4
        )
        logger.info("\n" + report)
        
        # Confusion matrix
        cm = confusion_matrix([l+1 for l in val_labels], [p+1 for p in val_preds])
        logger.info("\nConfusion Matrix:")
        logger.info(str(cm))
    
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE")
    logger.info("="*60)


if __name__ == "__main__":
    main()
