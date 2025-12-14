import json
import time
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformers import AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import joblib
import warnings
warnings.filterwarnings('ignore')

from config import *
from utils import setup_logger, set_seed
from models import LegalTextDataset, HuBERTClassifier

logger = setup_logger("training")

set_seed(RANDOM_SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

def train_baseline_model(train_df, val_df):
    print("TRAINING BASELINE MODEL (TF-IDF + Logistic Regression)")
    
    vectorizer = TfidfVectorizer(
        max_features=BASELINE_MAX_FEATURES,
        ngram_range=BASELINE_NGRAM_RANGE,
        min_df=2,
        max_df=0.95
    )
    
    X_train = vectorizer.fit_transform(train_df['text'])
    X_val = vectorizer.transform(val_df['text'])
    y_train = train_df['label'].values - 1
    y_val = val_df['label'].values - 1
    
    clf = LogisticRegression(
        max_iter=1000,
        C=1.0,
        class_weight='balanced',
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    
    clf.fit(X_train, y_train)
    
    train_pred = clf.predict(X_train)
    val_pred = clf.predict(X_val)
    
    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)
    train_f1 = f1_score(y_train, train_pred, average='weighted')
    val_f1 = f1_score(y_val, val_pred, average='weighted')
    
    print(f"\nBaseline results:")
    print(f"  Train accuracy: {train_acc:.4f} | Train F1: {train_f1:.4f}")
    print(f"  Val accuracy:   {val_acc:.4f} | Val F1:   {val_f1:.4f}")
    
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, MODEL_DIR / "baseline_vectorizer.pkl")
    joblib.dump(clf, MODEL_DIR / "baseline_model.pkl")
    
    return {
        'train_acc': train_acc,
        'val_acc': val_acc,
        'train_f1': train_f1,
        'val_f1': val_f1
    }

def train_epoch(model, dataloader, criterion, optimizer, scheduler, device, accumulation_steps, scaler=None):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    optimizer.zero_grad()
    
    progress_bar = tqdm(dataloader, desc="Training", ascii=True, ncols=100)
    for idx, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)
        
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels) / accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (idx + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_VALUE)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
        else:
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels) / accumulation_steps
            loss.backward()
            
            if (idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_VALUE)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
        
        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        progress_bar.set_postfix({'loss': loss.item() * accumulation_steps})
        
        if (idx + 1) % LOG_INTERVAL == 0:
            print(f"  Batch {idx+1}/{len(dataloader)} | Loss: {loss.item() * accumulation_steps:.4f}")
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return avg_loss, accuracy, f1


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", ascii=True, ncols=100):
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)
            
            if torch.cuda.is_available() and USE_MIXED_PRECISION:
                with torch.cuda.amp.autocast():
                    logits = model(input_ids, attention_mask)
                    loss = criterion(logits, labels)
            else:
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
            
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return avg_loss, accuracy, f1, all_preds, all_labels

def main():
    print("Legal text difficulty classification - training")
    
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\nLoading preprocessed data...")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "val.csv")
    
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")
    
    baseline_results = train_baseline_model(train_df, val_df)
    
    print(f"\nLoading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    print("Creating datasets...")
    train_dataset = LegalTextDataset(
        train_df['text'].values,
        train_df['label'].values,
        tokenizer
    )
    val_dataset = LegalTextDataset(
        val_df['text'].values,
        val_df['label'].values,
        tokenizer
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"\nInitializing model: {MODEL_NAME}")
    model = HuBERTClassifier().to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    
    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    total_steps = len(train_loader) * EPOCHS // ACCUMULATION_STEPS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=total_steps - warmup_steps,
        T_mult=1,
        eta_min=1e-7
    )
    
    scaler = torch.cuda.amp.GradScaler() if (USE_MIXED_PRECISION and torch.cuda.is_available()) else None
    if scaler:
        print("Mixed precision training enabled (FP16)")
    
    print(f"\nStarting training for {EPOCHS} epochs...")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Gradient accumulation steps: {ACCUMULATION_STEPS}")
    print(f"Effective batch size: {BATCH_SIZE * ACCUMULATION_STEPS}")
    
    best_val_f1 = 0
    patience_counter = 0
    training_history = []
    
    for epoch in range(EPOCHS):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, ACCUMULATION_STEPS, scaler
        )
        
        val_loss, val_acc, val_f1, _, _ = evaluate(
            model, val_loader, criterion, device
        )
        
        epoch_time = time.time() - start_time
        
        print(f"\nEpoch {epoch + 1} Results:")
        print(f"  Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} | Train F1: {train_f1:.4f}")
        print(f"  Val loss:   {val_loss:.4f} | Val acc:   {val_acc:.4f} | Val F1:   {val_f1:.4f}")
        print(f"  Time: {epoch_time:.2f}s")
        
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'train_f1': train_f1,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_f1': val_f1,
            'time': epoch_time
        })
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_acc': val_acc,
            }, MODEL_DIR / "best_model.pt")
            
            print(f" New best model saved! (Val F1: {val_f1:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")
            
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
    
    history_df = pd.DataFrame(training_history)
    history_df.to_csv(MODEL_DIR / "training_history.csv", index=False)
    
    results = {
        'baseline': baseline_results,
        'hubert': {
            'best_val_f1': best_val_f1,
            'total_epochs': len(training_history),
            'model_name': MODEL_NAME,
            'total_params': total_params,
            'trainable_params': trainable_params
        },
        'config': {
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'epochs': EPOCHS,
            'dropout_rate': DROPOUT_RATE,
            'weight_decay': WEIGHT_DECAY,
            'label_smoothing': LABEL_SMOOTHING,
            'max_length': MAX_LENGTH
        }
    }
    
    with open(MODEL_DIR / "training_results.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("Training completed!")
    print("="*60)
    print(f"\nBest validation F1: {best_val_f1:.4f}")
    print(f"Models saved to: {MODEL_DIR}")
    print(f"Training history saved to: {MODEL_DIR / 'training_history.csv'}")


if __name__ == "__main__":
    main()
