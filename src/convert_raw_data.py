import json
from pathlib import Path
import pandas as pd

RAW_DIR = Path("/data/raw/legaltextdecoder")
PROCESSED_DIR = Path("/data/processed")
EXCLUDE_FOLDERS = ["consensus"]

EVAL_FILE = RAW_DIR / "consensus" / "I1TLYH.json"

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
                print(f"WARNING: Ismeretlen cimke: '{label_text}' ({json_path.name}), skip")
                continue
            
            parsed.append({
                'text': text,
                'label': label
            })
            
        except Exception as e:
            print(f"Hiba ({json_path.name}): {e}")
            continue
    
    return parsed

def process_neptun_folders():  
    if not RAW_DIR.exists():
        print(f"Hiba: {RAW_DIR} nem letezik!")
        return None
    
    # Mappak keresese (consensus kihagyva)
    all_folders = [f for f in RAW_DIR.iterdir() if f.is_dir()]
    neptun_folders = [f for f in all_folders if f.name not in EXCLUDE_FOLDERS]
    
    print(f"Talalt mappak: {len(all_folders)}")
    print(f"Kihagyott: {EXCLUDE_FOLDERS}")
    print(f"Feldolgozando: {len(neptun_folders)}\n")
    
    all_data = []
    
    for folder in neptun_folders:
        json_files = list(folder.glob("*.json"))
        folder_data = []
        
        for json_file in json_files:
            parsed = parse_label_studio_json(json_file)
            folder_data.extend(parsed)
        
        all_data.extend(folder_data)
        print(f"   {folder.name}: {len(folder_data)} minta ({len(json_files)} JSON)")
    
    print(f"\nOsszes NEPTUN minta: {len(all_data)}")
    
    if not all_data:
        print("Nincs adat!")
        return None
    
    df = pd.DataFrame(all_data)
    
    initial = len(df)
    df = df.drop_duplicates(subset=['text'], keep='first')
    print(f"Duplikatumok: {initial - len(df)}")
    
    print(f"Vegso NEPTUN adat: {len(df)} minta\n")
    
    return df

def process_evaluation_file():
    if not EVAL_FILE.exists():
        print(f"Evaluation fajl nem talalhato: {EVAL_FILE}")
        return None
    
    print(f"Fajl: {EVAL_FILE.name}")
    
    # Parse
    data = parse_label_studio_json(EVAL_FILE)
    
    print(f"Mintak: {len(data)}")
    
    if not data:
        print("Nincs adat!")
        return None
    
    df = pd.DataFrame(data)
    
    initial = len(df)
    df = df.drop_duplicates(subset=['text'], keep='first')
    print(f"Duplikatumok: {initial - len(df)}")
    
    print(f"Vegso evaluation adat: {len(df)} minta\n")
    
    return df

def export_data(neptun_df, eval_df):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    if neptun_df is not None:
        neptun_csv = PROCESSED_DIR / "neptun_data.csv"
        neptun_df.to_csv(neptun_csv, index=False, encoding='utf-8')
        print(f"{neptun_csv} ({len(neptun_df)} minta)")
        
        neptun_json = PROCESSED_DIR / "neptun_data.json"
        data = neptun_df.to_dict(orient='records')
        with open(neptun_json, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"{neptun_json}")
    
    if eval_df is not None:
        eval_csv = PROCESSED_DIR / "evaluation.csv"
        eval_df.to_csv(eval_csv, index=False, encoding='utf-8')
        print(f"OK {eval_csv} ({len(eval_df)} minta)")
        
        eval_json = PROCESSED_DIR / "evaluation.json"
        data = eval_df.to_dict(orient='records')
        with open(eval_json, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"{eval_json}")

    print()

def print_summary(neptun_df, eval_df):
    if neptun_df is not None:
        print(f"NEPTUN adatok (train+test source):")
        print(f"Mintak: {len(neptun_df)}")
        print(f"Fajl:   neptun_data.csv, neptun_data.json")
        print()
        print(f"Osztalyeloszlas:")
        for label, count in neptun_df['label'].value_counts().sort_index().items():
            percentage = (count / len(neptun_df)) * 100
            print(f"Osztaly {label}: {count:3d} minta ({percentage:5.1f}%)")
        print()
    
    if eval_df is not None:
        print(f"Evaluation adatok (consensus):")
        print(f"Mintak: {len(eval_df)}")
        print(f"Fajl:   evaluation.csv, evaluation.json")
        print()
        print(f"Osztalyeloszlas:")
        for label, count in eval_df['label'].value_counts().sort_index().items():
            percentage = (count / len(eval_df)) * 100
            print(f"Osztaly {label}: {count:3d} minta ({percentage:5.1f}%)")
        print()
    
    print("Kovetkezo lepesek:")
    print("1. neptun_data.csv -> Train/Test split-eles")
    print("2. evaluation.csv -> Vegso kiertekeleshez")
def main():
    try:
        neptun_df = process_neptun_folders()
        eval_df = process_evaluation_file()
        export_data(neptun_df, eval_df)
        print_summary(neptun_df, eval_df)
        
    except Exception as e:
        print(f"\nHiba: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())