#!/bin/bash
set -e

wget -q --show-progress "https://bmeedu-my.sharepoint.com/:u:/g/personal/gyires-toth_balint_vik_bme_hu/IQDYwXUJcB_jQYr0bDfNT5RKARYgfKoH97zho3rxZ46KA1I?e=iFp3iz&download=1" -O /data/raw.zip
unzip -o /data/raw.zip -d /data/raw

python convert_raw_data.py

echo "Running data exploration notebook..."
jupyter nbconvert --to notebook --execute notebook/01-data-exploration.ipynb --output-dir /app/output --output 01-data-exploration-output

echo "Running label analysis notebook..."
jupyter nbconvert --to notebook --execute notebook/02-label-analysis.ipynb --output-dir /app/output --output 02-label-analysis-output

echo "Notebooks executed successfully!"

echo "Running data preprocessing..."
python 01-data-preprocessing.py

echo "Running model training..."
python 02-training.py

echo "Running model evaluation..."
python 03-evaluation.py

echo "Running inference demo on unseen data (E77YIW)..."

python 04-inference.py --mode file --input /data/processed/inference_demo.csv --output /data/output/inference_predictions.csv --model /data/output/models/best_model.pt --batch-size 32