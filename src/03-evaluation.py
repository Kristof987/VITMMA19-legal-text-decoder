import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score,
    classification_report, 
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import joblib
warnings.filterwarnings('ignore')

from config import *
from utils import setup_logger
from models import LegalTextDataset, HuBERTClassifier

logger = setup_logger("evaluation")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")
print(f"Using device: {device}")


def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)
            
            if torch.cuda.is_available() and USE_MIXED_PRECISION:
                with torch.cuda.amp.autocast():
                    logits = model(input_ids, attention_mask)
            else:
                logits = model(input_ids, attention_mask)
            
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def compute_metrics(y_true, y_pred):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0)
    }
    
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    
    metrics['per_class'] = {
        'f1': per_class_f1.tolist(),
        'precision': per_class_precision.tolist(),
        'recall': per_class_recall.tolist()
    }
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, labels, save_path):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion matrix', fontsize=16, pad=20)
    plt.ylabel('True label', fontsize=12)
    plt.xlabel('Predicted label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to {save_path}")


def plot_per_class_metrics(metrics, labels, save_path):
    per_class = metrics['per_class']
    
    x = np.arange(len(labels))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.bar(x - width, per_class['precision'], width, label='Precision', alpha=0.8)
    ax.bar(x, per_class['recall'], width, label='Recall', alpha=0.8)
    ax.bar(x + width, per_class['f1'], width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('Difficulty level', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-class metrics', fontsize=16, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Per-class metrics plot saved to {save_path}")


def evaluate_baseline(test_df):
    print("\n" + "="*60)
    print("EVALUATING BASELINE MODEL")
    print("="*60)
    
    vectorizer = joblib.load(MODEL_DIR / "baseline_vectorizer.pkl")
    clf = joblib.load(MODEL_DIR / "baseline_model.pkl")
    
    X_test = vectorizer.transform(test_df['text'])
    y_test = test_df['label'].values - 1
    
    y_pred = clf.predict(X_test)
    
    metrics = compute_metrics(y_test, y_pred)
    
    print(f"\nBaseline Results:")
    print(f"  Accuracy:          {metrics['accuracy']:.4f}")
    print(f"  F1-Score (Macro):  {metrics['f1_macro']:.4f}")
    print(f"  F1-Score (Weighted): {metrics['f1_weighted']:.4f}")
    print(f"  Precision (Macro): {metrics['precision_macro']:.4f}")
    print(f"  Recall (Macro):    {metrics['recall_macro']:.4f}")
    
    return metrics, y_pred, y_test


def evaluate_hubert(test_df):
    print("\n" + "="*60)
    print("EVALUATING HUBERT MODEL")
    print("="*60)
    
    print(f"\nLoading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    test_dataset = LegalTextDataset(
        test_df['text'].values,
        test_df['label'].values,
        tokenizer
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"Loading best model from {MODEL_DIR / 'best_model.pt'}")
    model = HuBERTClassifier().to(device)
    
    checkpoint = torch.load(MODEL_DIR / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Model trained for {checkpoint['epoch']} epochs")
    print(f"Best validation F1: {checkpoint['val_f1']:.4f}")
    
    print("\nEvaluating on test set...")
    y_pred, y_test, y_probs = evaluate_model(model, test_loader, device)
    
    metrics = compute_metrics(y_test, y_pred)
    
    print(f"\nHuBERT Results:")
    print(f"  Accuracy:          {metrics['accuracy']:.4f}")
    print(f"  F1-Score (Macro):  {metrics['f1_macro']:.4f}")
    print(f"  F1-Score (Weighted): {metrics['f1_weighted']:.4f}")
    print(f"  Precision (Macro): {metrics['precision_macro']:.4f}")
    print(f"  Recall (Macro):    {metrics['recall_macro']:.4f}")
    
    return metrics, y_pred, y_test, y_probs

def main():
    print("Legal text difficulty classification - evaluation")
    
    eval_dir = Path("/data/output/evaluation")
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nLoading test data...")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    print(f"Test samples: {len(test_df)}")
    
    print(f"\nTest label distribution:")
    for label in sorted(test_df['label'].unique()):
        count = (test_df['label'] == label).sum()
        pct = (count / len(test_df)) * 100
        print(f"  Label {label} ({LABEL_NAMES[label]}): {count:4d} ({pct:5.1f}%)")
    
    baseline_metrics, baseline_pred, y_test = evaluate_baseline(test_df)
    
    hubert_metrics, hubert_pred, y_test, hubert_probs = evaluate_hubert(test_df)
    
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    comparison = pd.DataFrame({
        'Metric': ['Accuracy', 'F1 (Macro)', 'F1 (Weighted)', 'Precision (Macro)', 'Recall (Macro)'],
        'Baseline': [
            baseline_metrics['accuracy'],
            baseline_metrics['f1_macro'],
            baseline_metrics['f1_weighted'],
            baseline_metrics['precision_macro'],
            baseline_metrics['recall_macro']
        ],
        'HuBERT': [
            hubert_metrics['accuracy'],
            hubert_metrics['f1_macro'],
            hubert_metrics['f1_weighted'],
            hubert_metrics['precision_macro'],
            hubert_metrics['recall_macro']
        ]
    })
    
    comparison['Improvement'] = comparison['HuBERT'] - comparison['Baseline']
    comparison['Improvement %'] = (comparison['Improvement'] / comparison['Baseline']) * 100
    
    print("\n" + comparison.to_string(index=False))
    
    comparison.to_csv(eval_dir / "model_comparison.csv", index=False)
    
    print("\n" + "="*60)
    print("Detailed classification reports")
    print("="*60)
    
    label_names_list = [LABEL_NAMES[i+1] for i in range(NUM_CLASSES)]
    
    print("\nBaseline Model:")
    print(classification_report(
        y_test, 
        baseline_pred, 
        target_names=label_names_list,
        digits=4
    ))
    
    print("\nHuBERT Model:")
    print(classification_report(
        y_test, 
        hubert_pred, 
        target_names=label_names_list,
        digits=4
    ))
    
    print("\n" + "="*60)
    print("Creating visualizations...")
    print("="*60)
    
    plot_confusion_matrix(
        y_test, 
        baseline_pred, 
        label_names_list,
        eval_dir / "confusion_matrix_baseline.png"
    )
    
    plot_confusion_matrix(
        y_test, 
        hubert_pred, 
        label_names_list,
        eval_dir / "confusion_matrix_hubert.png"
    )
    
    plot_per_class_metrics(
        baseline_metrics,
        label_names_list,
        eval_dir / "per_class_metrics_baseline.png"
    )
    
    plot_per_class_metrics(
        hubert_metrics,
        label_names_list,
        eval_dir / "per_class_metrics_hubert.png"
    )
    
    results = {
        'test_samples': len(test_df),
        'baseline': {
            'accuracy': float(baseline_metrics['accuracy']),
            'f1_macro': float(baseline_metrics['f1_macro']),
            'f1_weighted': float(baseline_metrics['f1_weighted']),
            'precision_macro': float(baseline_metrics['precision_macro']),
            'recall_macro': float(baseline_metrics['recall_macro']),
            'per_class_f1': [float(x) for x in baseline_metrics['per_class']['f1']]
        },
        'hubert': {
            'accuracy': float(hubert_metrics['accuracy']),
            'f1_macro': float(hubert_metrics['f1_macro']),
            'f1_weighted': float(hubert_metrics['f1_weighted']),
            'precision_macro': float(hubert_metrics['precision_macro']),
            'recall_macro': float(hubert_metrics['recall_macro']),
            'per_class_f1': [float(x) for x in hubert_metrics['per_class']['f1']]
        },
        'improvement': {
            'accuracy': float(hubert_metrics['accuracy'] - baseline_metrics['accuracy']),
            'f1_macro': float(hubert_metrics['f1_macro'] - baseline_metrics['f1_macro']),
            'f1_weighted': float(hubert_metrics['f1_weighted'] - baseline_metrics['f1_weighted'])
        }
    }
    
    with open(eval_dir / "evaluation_results.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to {eval_dir / 'evaluation_results.json'}")
    
    predictions_df = pd.DataFrame({
        'text': test_df['text'].values,
        'true_label': y_test + 1,
        'baseline_pred': baseline_pred + 1,
        'hubert_pred': hubert_pred + 1,
        'true_label_name': [LABEL_NAMES[i+1] for i in y_test],
        'baseline_pred_name': [LABEL_NAMES[i+1] for i in baseline_pred],
        'hubert_pred_name': [LABEL_NAMES[i+1] for i in hubert_pred]
    })
    
    for i in range(NUM_CLASSES):
        predictions_df[f'hubert_prob_class_{i+1}'] = hubert_probs[:, i]
    
    predictions_df.to_csv(eval_dir / "predictions.csv", index=False, encoding='utf-8')
    print(f"Predictions saved to {eval_dir / 'predictions.csv'}")
    
    print("\n" + "="*60)
    print("Evaluation completed!")
    print("="*60)
    print(f"\nResults directory: {eval_dir}")
    print(f"\nKey findings:")
    print(f"  Baseline Accuracy: {baseline_metrics['accuracy']:.2%}")
    print(f"  HuBERT Accuracy:   {hubert_metrics['accuracy']:.2%}")
    print(f"  Improvement:       {results['improvement']['accuracy']:.2%}")


if __name__ == "__main__":
    main()
