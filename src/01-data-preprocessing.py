"""
TODO: leírás
"""

import pandas as pd
import numpy as np
import json
import re
from pathlib import Path
from sklearn.model_selection import train_test_split

from config import RANDOM_SEED, TEST_SIZE, VAL_SIZE

PROCESSED_DIR = Path("/data/processed")
OUTPUT_DIR = Path("/data/output")

np.random.seed(RANDOM_SEED)


def clean_text(text):
    """
    Clean and normalize text data.
    """

    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    text = text.replace('\xa0', ' ')
    
    return text


def validate_data(df, dataset_name):
    """
    Validate dataset for quality issues.
    """
    print(f"Validating {dataset_name}")
    
    missing = df.isnull().sum()
    if missing.any():
        print(f"WARNING: Missing values found:")
        print(missing[missing > 0])
    else:
        print("No missing values")
    
    empty_texts = df['text'].str.strip().str.len() == 0
    if empty_texts.any():
        print(f"WARNING: {empty_texts.sum()} empty texts found")
        df = df[~empty_texts].copy()
    else:
        print("No empty texts")
    
    duplicates = df.duplicated(subset=['text'])
    if duplicates.any():
        print(f"INFO: {duplicates.sum()} duplicate texts found")
    else:
        print("No duplicates")
    
    print(f"\nLabel distribution:")
    label_counts = df['label'].value_counts().sort_index()
    for label, count in label_counts.items():
        pct = (count / len(df)) * 100
        print(f"  Label {label}: {count:4d} ({pct:5.1f}%)")
    
    text_lengths = df['text'].str.len()
    print(f"\nText length statistics:")
    print(f"  Min: {text_lengths.min()}")
    print(f"  Max: {text_lengths.max()}")
    print(f"  Mean: {text_lengths.mean():.1f}")
    print(f"  Median: {text_lengths.median():.1f}")
    
    return df


def augment_text(text, augmentation_type='synonym'):
    """
    Simple text augmentation for regularization.
    
    Data augmentation is a regularization technique that helps prevent overfitting
    by creating variations of training samples.
    
    Args:
        text: Input text
        augmentation_type: Type of augmentation ('synonym', 'swap', 'delete')
        
    Returns:
        Augmented text
    """
    words = text.split()
    
    if len(words) < 3:
        return text
    
    if augmentation_type == 'swap':
        if len(words) >= 2:
            idx1, idx2 = np.random.choice(len(words), 2, replace=False)
            words[idx1], words[idx2] = words[idx2], words[idx1]
    
    elif augmentation_type == 'delete':
        num_to_delete = max(1, int(len(words) * 0.1))
        indices_to_delete = np.random.choice(len(words), num_to_delete, replace=False)
        words = [w for i, w in enumerate(words) if i not in indices_to_delete]
    
    return ' '.join(words)


def create_augmented_samples(df, augmentation_ratio=0.2, min_samples_per_class=200):
    """
    Create augmented samples for minority classes to balance the dataset.
    
    This is a regularization technique that helps with class imbalance.
    
    Args:
        df: Input DataFrame
        augmentation_ratio: Ratio of samples to augment
        min_samples_per_class: Minimum samples per class before augmentation
        
    Returns:
        DataFrame with augmented samples
    """
    augmented_samples = []
    
    for label in df['label'].unique():
        label_df = df[df['label'] == label]
        
        if len(label_df) < min_samples_per_class:
            num_to_augment = int(len(label_df) * augmentation_ratio)
            
            samples_to_augment = label_df.sample(n=num_to_augment, replace=True, random_state=RANDOM_SEED)
            
            for _, row in samples_to_augment.iterrows():
                aug_type = np.random.choice(['swap', 'delete'])
                augmented_text = augment_text(row['text'], aug_type)
                
                augmented_samples.append({
                    'text': augmented_text,
                    'label': row['label']
                })
    
    if augmented_samples:
        aug_df = pd.DataFrame(augmented_samples)
        print(f"\nCreated {len(aug_df)} augmented samples")
        return pd.concat([df, aug_df], ignore_index=True)
    
    return df


def stratified_split(df, test_size, val_size, random_state=42):
    """
    Perform stratified train/validation/test split.
    
    Stratified splitting ensures that the label distribution is preserved
    across all splits, which is important for:
    - Avoiding overfitting (proper validation)
    - Reliable performance estimation
    - Handling class imbalance
    
    Args:
        df: Input DataFrame
        test_size: Proportion for test set
        val_size: Proportion for validation set (from remaining after test)
        random_state: Random seed for reproducibility
        
    Returns:
        train_df, val_df, test_df
    """
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df['label'],
        random_state=random_state
    )
    
    val_size_adjusted = val_size / (1 - test_size)
    
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size_adjusted,
        stratify=train_val_df['label'],
        random_state=random_state
    )
    
    return train_df, val_df, test_df



def compute_statistics(train_df, val_df, test_df):
    """
    Compute and display statistics for all splits.
    
    Args:
        train_df, val_df, test_df: DataFrames for each split
    """
    print("Dataset split statistics")
    
    for name, df in [('Train', train_df), ('Validation', val_df), ('Test', test_df)]:
        print(f"\n{name} Set:")
        print(f"  Total samples: {len(df)}")
        print(f"  Label distribution:")
        
        label_counts = df['label'].value_counts().sort_index()
        for label, count in label_counts.items():
            pct = (count / len(df)) * 100
            print(f"    Label {label}: {count:4d} ({pct:5.1f}%)")
        
        text_lengths = df['text'].str.len()
        print(f"  Text length: mean={text_lengths.mean():.1f}, "
              f"median={text_lengths.median():.1f}, "
              f"std={text_lengths.std():.1f}")


def save_splits(train_df, val_df, test_df, output_dir):
    """
    Save train/val/test splits to disk.
    
    Args:
        train_df, val_df, test_df: DataFrames to save
        output_dir: Output directory path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_df.to_csv(output_dir / "train.csv", index=False, encoding='utf-8')
    val_df.to_csv(output_dir / "val.csv", index=False, encoding='utf-8')
    test_df.to_csv(output_dir / "test.csv", index=False, encoding='utf-8')
    
    train_df.to_json(output_dir / "train.json", orient='records', force_ascii=False, indent=2)
    val_df.to_json(output_dir / "val.json", orient='records', force_ascii=False, indent=2)
    test_df.to_json(output_dir / "test.json", orient='records', force_ascii=False, indent=2)
    
    print("Saved preprocessed data:")
    print(f"  {output_dir / 'train.csv'} ({len(train_df)} samples)")
    print(f"  {output_dir / 'val.csv'} ({len(val_df)} samples)")
    print(f"  {output_dir / 'test.csv'} ({len(test_df)} samples)")

def main():
    """
    Main preprocessing pipeline.
    
    Pipeline steps:
    1. Load processed data from neptun_data.csv
    2. Validate and clean data
    3. Apply text augmentation for regularization
    4. Perform stratified train/val/test split
    5. Save preprocessed splits
    """
    print("\nDATA PREPROCESSING PIPELINE")

    print("\nLoading processed data...")
    neptun_df = pd.read_csv(PROCESSED_DIR / "neptun_data.csv")
    print(f"Loaded {len(neptun_df)} samples from Train dataset")
    
    neptun_df = validate_data(neptun_df, "Train Dataset")
    
    print("\nCleaning text data...")
    neptun_df['text'] = neptun_df['text'].apply(clean_text)
    
    print("\nApplying data augmentation for regularization...")
    neptun_df = create_augmented_samples(
        neptun_df,
        augmentation_ratio=0.15,
        min_samples_per_class=250
    )
    
    print(f"\nPerforming stratified split...")
    print(f"  Test size: {TEST_SIZE*100:.0f}%")
    print(f"  Validation size: {VAL_SIZE*100:.0f}%")
    print(f"  Train size: {(1-TEST_SIZE-VAL_SIZE)*100:.0f}%")
    
    train_df, val_df, test_df = stratified_split(
        neptun_df,
        test_size=TEST_SIZE,
        val_size=VAL_SIZE,
        random_state=RANDOM_SEED
    )
    
    compute_statistics(train_df, val_df, test_df)
    
    save_splits(train_df, val_df, test_df, PROCESSED_DIR)

    metadata = {
        'total_samples': len(neptun_df),
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'test_samples': len(test_df),
        'test_size': TEST_SIZE,
        'val_size': VAL_SIZE,
        'random_seed': RANDOM_SEED,
        'num_classes': int(neptun_df['label'].nunique()),
        'label_names': {
            1: 'Very Hard',
            2: 'Hard',
            3: 'Moderate',
            4: 'Easy',
            5: 'Very Easy'
        }
    }
    
    with open(PROCESSED_DIR / "metadata.json", 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\nMetadata saved to {PROCESSED_DIR / 'metadata.json'}")
    print("\nPreprocessing complete!")


if __name__ == "__main__":
    main()
