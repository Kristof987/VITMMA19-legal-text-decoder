import json
from pathlib import Path
import pandas as pd

RAW_DIR = Path("/data/raw/legaltextdecoder")
PROCESSED_DIR = Path("/data/processed")
EXCLUDE_FOLDERS = ["consensus"]

# Evaluation file (consensus data)
EVAL_FILE = RAW_DIR / "consensus" / "I1TLYH.json"

# Inference demo file (unseen data)
INFERENCE_FILE = RAW_DIR / "E77YIW" / "mak_aszf_cimkezes.json"

# Files to exclude from training
EXCLUDE_FILES = [
    "mak_aszf_cimkezes.json"    # E77YIW folder (used for inference)
]

LABEL_MAPPING = {
    "1-Nagyon nehezen érthető": 1,
    "2-Nehezen érthető": 2,
    "3-Többé/kevésbé megértem": 3,
    "4-Érthető": 4,
    "5-Könnyen érthető": 5
}

def parse_label_studio_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        items = data
    else:
        items = [data]
    
    parsed = []
    
    for item in items:
        try:
            text = item.get('data', {}).get('text', '').strip()
            if not text:
                continue
            
            annotations = item.get('annotations', [])
            if not annotations:
                continue
            
            result = annotations[0].get('result', [])
            if not result:
                continue
            
            choices = result[0].get('value', {}).get('choices', [])
            if not choices:
                continue
            
            label_text = choices[0]
            
            label = LABEL_MAPPING.get(label_text)
            if label is None:
                print(f"WARNING: Unknown label: '{label_text}' ({json_path.name}), skip")
                continue
            
            parsed.append({
                'text': text,
                'label': label
            })
            
        except Exception as e:
            print(f"Error ({json_path.name}): {e}")
            continue
    
    return parsed

def process_neptun_folders():  
    if not RAW_DIR.exists():
        print(f"Error: {RAW_DIR} does not exist!")
        return None
    
    all_folders = [f for f in RAW_DIR.iterdir() if f.is_dir()]
    neptun_folders = [f for f in all_folders if f.name not in EXCLUDE_FOLDERS]
    
    print(f"Found folders: {len(all_folders)}")
    print(f"Skipped: {EXCLUDE_FOLDERS}")
    print(f"To be processed: {len(neptun_folders)}\n")
    
    all_data = []
    
    for folder in neptun_folders:
        json_files = list(folder.glob("*.json"))
        
        # Filter out excluded files
        json_files = [f for f in json_files if f.name not in EXCLUDE_FILES]
        
        folder_data = []
        
        for json_file in json_files:
            parsed = parse_label_studio_json(json_file)
            folder_data.extend(parsed)
        
        all_data.extend(folder_data)
        print(f"   {folder.name}: {len(folder_data)} sample ({len(json_files)} JSON)")
    
    print(f"\nAll train sample: {len(all_data)}")
    
    if not all_data:
        print("No data found")
        return None
    
    df = pd.DataFrame(all_data)
    
    initial = len(df)
    df = df.drop_duplicates(subset=['text'], keep='first')
    print(f"Duplicates: {initial - len(df)}")
    
    print(f"Final train data: {len(df)} sample\n")
    
    return df

def process_evaluation_file():
    """Process consensus evaluation file (I1TLYH.json)"""
    if not EVAL_FILE.exists():
        print(f"Evaluation file not found: {EVAL_FILE}")
        return None
    
    print(f"File: {EVAL_FILE.name}")
    
    data = parse_label_studio_json(EVAL_FILE)
    
    print(f"Samples: {len(data)}")
    
    if not data:
        print("No data found!")
        return None
    
    df = pd.DataFrame(data)
    
    initial = len(df)
    df = df.drop_duplicates(subset=['text'], keep='first')
    print(f"Duplicates: {initial - len(df)}")
    
    print(f"Final evaluation data: {len(df)} sample\n")
    
    return df


def process_inference_file():
    """Process inference demo file (E77YIW/mak_aszf_cimkezes.json)"""
    if not INFERENCE_FILE.exists():
        print(f"Inference file not found: {INFERENCE_FILE}")
        return None
    
    print(f"File: {INFERENCE_FILE.name}")
    
    data = parse_label_studio_json(INFERENCE_FILE)
    
    print(f"Samples: {len(data)}")
    
    if not data:
        print("No data found!")
        return None
    
    df = pd.DataFrame(data)
    
    initial = len(df)
    df = df.drop_duplicates(subset=['text'], keep='first')
    print(f"Duplicates: {initial - len(df)}")
    
    print(f"Final inference data: {len(df)} sample\n")
    
    return df

def export_data(neptun_df, eval_df, inference_df):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    if neptun_df is not None:
        neptun_csv = PROCESSED_DIR / "neptun_data.csv"
        neptun_df.to_csv(neptun_csv, index=False, encoding='utf-8')
        print(f"{neptun_csv} ({len(neptun_df)} sample)")
        
        neptun_json = PROCESSED_DIR / "neptun_data.json"
        data = neptun_df.to_dict(orient='records')
        with open(neptun_json, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"{neptun_json}")
    
    if eval_df is not None:
        eval_csv = PROCESSED_DIR / "evaluation.csv"
        eval_df.to_csv(eval_csv, index=False, encoding='utf-8')
        print(f"OK {eval_csv} ({len(eval_df)} sample)")
        
        eval_json = PROCESSED_DIR / "evaluation.json"
        data = eval_df.to_dict(orient='records')
        with open(eval_json, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"{eval_json}")
    
    if inference_df is not None:
        inference_csv = PROCESSED_DIR / "inference_demo.csv"
        inference_df.to_csv(inference_csv, index=False, encoding='utf-8')
        print(f"OK {inference_csv} ({len(inference_df)} sample)")
        
        inference_json = PROCESSED_DIR / "inference_demo.json"
        data = inference_df.to_dict(orient='records')
        with open(inference_json, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"{inference_json}")

    print()

def print_summary(neptun_df, eval_df, inference_df):
    print("="*60)
    print("DATA PROCESSING SUMMARY")
    print("="*60)
    
    if neptun_df is not None:
        print(f"\nTRAIN DATA (NEPTUN):")
        print(f"  Samples: {len(neptun_df)}")
        print(f"  Files:   neptun_data.csv, neptun_data.json")
        print(f"\n  Class distribution:")
        for label, count in neptun_df['label'].value_counts().sort_index().items():
            percentage = (count / len(neptun_df)) * 100
            print(f"    Class {label}: {count:3d} sample ({percentage:5.1f}%)")
    
    if eval_df is not None:
        print(f"\nEVALUATION DATA (Consensus - I1TLYH):")
        print(f"  Samples: {len(eval_df)}")
        print(f"  Files:   evaluation.csv, evaluation.json")
        print(f"\n  Class distribution:")
        for label, count in eval_df['label'].value_counts().sort_index().items():
            percentage = (count / len(eval_df)) * 100
            print(f"    Class {label}: {count:3d} sample ({percentage:5.1f}%)")
    
    if inference_df is not None:
        print(f"\nINFERENCE DEMO DATA (E77YIW):")
        print(f"  Samples: {len(inference_df)}")
        print(f"  Files:   inference_demo.csv, inference_demo.json")
        print(f"\n  Class distribution:")
        for label, count in inference_df['label'].value_counts().sort_index().items():
            percentage = (count / len(inference_df)) * 100
            print(f"    Class {label}: {count:3d} sample ({percentage:5.1f}%)")
    
    print("\n" + "="*60)
    print("Next steps:")
    print("  1. neptun_data.csv -> Train/Val/Test split")
    print("  2. evaluation.csv -> Final model evaluation")
    print("  3. inference_demo.csv -> Inference demonstration")
    print("="*60)

def main():
    try:
        print("="*60)
        print("PROCESSING RAW DATA")
        print("="*60)
        print(f"\nExcluded files: {EXCLUDE_FILES}\n")
        
        neptun_df = process_neptun_folders()
        eval_df = process_evaluation_file()
        inference_df = process_inference_file()
        
        export_data(neptun_df, eval_df, inference_df)
        print_summary(neptun_df, eval_df, inference_df)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())