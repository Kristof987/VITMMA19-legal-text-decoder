"""
Inference script for legal text difficulty classification.
Runs predictions on new, unseen legal texts using the trained HuBERT model.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import config
from config import *

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# DATASET CLASS
# ============================================================================

class InferenceDataset(Dataset):
    """PyTorch Dataset for inference (no labels)."""
    
    def __init__(self, texts, tokenizer, max_length=MAX_LENGTH):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'text': text
        }


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class HuBERTClassifier(nn.Module):
    """HuBERT-based text classifier."""
    
    def __init__(self, model_name=MODEL_NAME, num_classes=NUM_CLASSES, 
                 dropout_rate=DROPOUT_RATE, hidden_size=CLASSIFIER_HIDDEN_SIZE):
        super(HuBERTClassifier, self).__init__()
        
        self.hubert = AutoModel.from_pretrained(model_name)
        
        # Classification head
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(HIDDEN_SIZE, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.hubert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        x = self.dropout1(pooled_output)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        logits = self.fc2(x)
        
        return logits


# ============================================================================
# INFERENCE FUNCTIONS
# ============================================================================

def load_model(model_path, device):
    """
    Load trained model from checkpoint.
    """
    print(f"Loading model from {model_path}")
    
    model = HuBERTClassifier().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Model loaded successfully")
    print(f"  Trained for {checkpoint['epoch']} epochs")
    print(f"  Best validation F1: {checkpoint['val_f1']:.4f}")
    
    return model


def predict_batch(model, dataloader, device):
    """
    Run inference on a batch of texts.
    
    Returns:
        predictions: Predicted labels (1-5)
        probabilities: Class probabilities for each sample
        texts: Original texts
    """
    model.eval()
    
    all_preds = []
    all_probs = []
    all_texts = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            texts = batch['text']
            
            # Mixed precision inference
            if torch.cuda.is_available() and USE_MIXED_PRECISION:
                with torch.cuda.amp.autocast():
                    logits = model(input_ids, attention_mask)
            else:
                logits = model(input_ids, attention_mask)
            
            # Get predictions and probabilities
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_texts.extend(texts)
    
    # Convert predictions from 0-4 to 1-5
    predictions = np.array(all_preds) + 1
    probabilities = np.array(all_probs)
    
    return predictions, probabilities, all_texts


def predict_single_text(model, tokenizer, text, device):
    """
    Run inference on a single text.
    
    Returns:
        prediction: Predicted label (1-5)
        probabilities: Class probabilities
        confidence: Confidence score (max probability)
    """
    model.eval()
    
    # Tokenize
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=MAX_LENGTH,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        if torch.cuda.is_available() and USE_MIXED_PRECISION:
            with torch.cuda.amp.autocast():
                logits = model(input_ids, attention_mask)
        else:
            logits = model(input_ids, attention_mask)
        
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(logits, dim=1)
    
    prediction = pred.item() + 1  # Convert 0-4 to 1-5
    probabilities = probs.cpu().numpy()[0]
    confidence = probabilities.max()
    
    return prediction, probabilities, confidence


def format_prediction_output(prediction, probabilities, text, verbose=True):
    """
    Format prediction output for display.
    """
    label_name = LABEL_NAMES[prediction]
    confidence = probabilities[prediction - 1]
    
    output = {
        'text': text[:200] + '...' if len(text) > 200 else text,
        'predicted_label': int(prediction),
        'predicted_difficulty': label_name,
        'confidence': float(confidence),
        'probabilities': {
            LABEL_NAMES[i+1]: float(probabilities[i]) 
            for i in range(NUM_CLASSES)
        }
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Text: {output['text']}")
        print(f"{'='*60}")
        print(f"Predicted Difficulty: {label_name} (Label {prediction})")
        print(f"Confidence: {confidence:.2%}")
        print(f"\nClass Probabilities:")
        for i in range(NUM_CLASSES):
            label = i + 1
            prob = probabilities[i]
            bar = '█' * int(prob * 50)
            print(f"  {label} ({LABEL_NAMES[label]:12s}): {prob:.2%} {bar}")
    
    return output


# ============================================================================
# MAIN INFERENCE FUNCTIONS
# ============================================================================

def inference_from_file(input_file, output_file, model_path, batch_size=32):
    """
    Run inference on texts from a CSV file.
    
    Args:
        input_file: Path to input CSV file (must have 'text' column)
        output_file: Path to save predictions
        model_path: Path to trained model checkpoint
        batch_size: Batch size for inference
    """
    print("="*60)
    print("LEGAL TEXT DIFFICULTY CLASSIFICATION - INFERENCE")
    print("="*60)
    print(f"\nDevice: {device}")
    
    # Load data
    print(f"\nLoading data from {input_file}")
    df = pd.read_csv(input_file)
    
    if 'text' not in df.columns:
        raise ValueError("Input CSV must have a 'text' column")
    
    print(f"✓ Loaded {len(df)} texts")
    
    # Load tokenizer
    print(f"\nLoading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Load model
    model = load_model(model_path, device)
    
    # Create dataset and dataloader
    dataset = InferenceDataset(df['text'].values, tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Run inference
    print(f"\nRunning inference on {len(df)} texts...")
    predictions, probabilities, texts = predict_batch(model, dataloader, device)
    
    # Create results dataframe
    results_df = df.copy()
    results_df['predicted_label'] = predictions
    results_df['predicted_difficulty'] = [LABEL_NAMES[p] for p in predictions]
    results_df['confidence'] = probabilities.max(axis=1)
    
    # Add probability columns
    for i in range(NUM_CLASSES):
        results_df[f'prob_label_{i+1}'] = probabilities[:, i]
    
    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"\n✓ Predictions saved to {output_path}")
    
    # Summary statistics
    print(f"\n{'='*60}")
    print("PREDICTION SUMMARY")
    print(f"{'='*60}")
    print(f"\nPrediction distribution:")
    for label in sorted(LABEL_NAMES.keys()):
        count = (predictions == label).sum()
        pct = (count / len(predictions)) * 100
        print(f"  {label} ({LABEL_NAMES[label]:12s}): {count:4d} ({pct:5.1f}%)")
    
    print(f"\nConfidence statistics:")
    print(f"  Mean confidence: {probabilities.max(axis=1).mean():.2%}")
    print(f"  Min confidence:  {probabilities.max(axis=1).min():.2%}")
    print(f"  Max confidence:  {probabilities.max(axis=1).max():.2%}")
    
    low_conf_threshold = 0.5
    low_conf_count = (probabilities.max(axis=1) < low_conf_threshold).sum()
    if low_conf_count > 0:
        print(f"\n{low_conf_count} predictions with confidence < {low_conf_threshold:.0%}")
    
    return results_df


def inference_interactive(model_path):
    """
    Interactive inference mode - enter texts manually.
    """
    print("="*60)
    print("INTERACTIVE INFERENCE MODE")
    print("="*60)
    print(f"\nDevice: {device}")
    
    # Load tokenizer and model
    print(f"\nLoading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    model = load_model(model_path, device)
    
    print("\n" + "="*60)
    print("Ready for inference!")
    print("Enter legal text to classify (or 'quit' to exit)")
    print("="*60)
    
    while True:
        print("\n" + "-"*60)
        text = input("\nEnter text: ").strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            print("\nExiting...")
            break
        
        if not text:
            print("⚠️  Please enter some text")
            continue
        
        # Run prediction
        prediction, probabilities, confidence = predict_single_text(
            model, tokenizer, text, device
        )
        
        # Display results
        format_prediction_output(prediction, probabilities, text, verbose=True)


def inference_from_text(text, model_path, verbose=True):
    """
    Run inference on a single text string.
    
    Args:
        text: Input text
        model_path: Path to trained model checkpoint
        verbose: Print detailed output
        
    Returns:
        Dictionary with prediction results
    """
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = load_model(model_path, device)
    
    # Run prediction
    prediction, probabilities, confidence = predict_single_text(
        model, tokenizer, text, device
    )
    
    # Format output
    return format_prediction_output(prediction, probabilities, text, verbose)


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Legal Text Difficulty Classification - Inference'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['file', 'interactive', 'text'],
        default='file',
        help='Inference mode: file (batch), interactive, or text (single)'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        help='Input CSV file path (for file mode)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output CSV file path (for file mode)'
    )
    
    parser.add_argument(
        '--text',
        type=str,
        help='Input text (for text mode)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default=str(MODEL_DIR / "best_model.pt"),
        help='Path to trained model checkpoint'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for inference (file mode)'
    )
    
    args = parser.parse_args()
    
    # Validate model path
    if not Path(args.model).exists():
        raise FileNotFoundError(f"Model not found: {args.model}")
    
    # Run inference based on mode
    if args.mode == 'file':
        if not args.input:
            raise ValueError("--input required for file mode")
        if not args.output:
            raise ValueError("--output required for file mode")
        
        inference_from_file(
            args.input,
            args.output,
            args.model,
            args.batch_size
        )
    
    elif args.mode == 'interactive':
        inference_interactive(args.model)
    
    elif args.mode == 'text':
        if not args.text:
            raise ValueError("--text required for text mode")
        
        inference_from_text(args.text, args.model, verbose=True)


if __name__ == "__main__":
    main()
