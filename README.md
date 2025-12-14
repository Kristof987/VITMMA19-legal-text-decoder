# Deep Learning Class (VITMMA19) Project Work

Hungarian legal text (Általános Szerződési Feltételek - ÁSZF) difficulty classification using fine-tuned HuBERT model.

## Submission Instructions

[Delete this entire section after reading and following the instructions.]

### Project Levels

**Basic Level (for signature)**
*   Containerization
*   Data acquisition and analysis
*   Data preparation
*   Baseline (reference) model
*   Model development
*   Basic evaluation

**Outstanding Level (aiming for +1 mark)**
*   Containerization
*   Data acquisition and analysis
*   Data cleansing and preparation
*   Defining evaluation criteria
*   Baseline (reference) model
*   Incremental model development
*   Advanced evaluation
*   ML as a service (backend) with GUI frontend
*   Creative ideas, well-developed solutions, and exceptional performance can also earn an extra grade (+1 mark).

### Data Preparation

**Important:** You must provide a script (or at least a precise description) of how to convert the raw database into a format that can be processed by the scripts.
* The scripts should ideally download the data from there or process it directly from the current sharepoint location.
* Or if you do partly manual preparation, then it is recommended to upload the prepared data format to a shared folder and access from there.

[Describe the data preparation process here]

### Logging Requirements

The training process must produce a log file that captures the following essential information for grading:

1.  **Configuration**: Print the hyperparameters used (e.g., number of epochs, batch size, learning rate).
2.  **Data Processing**: Confirm successful data loading and preprocessing steps.
3.  **Model Architecture**: A summary of the model structure with the number of parameters (trainable and non-trainable).
4.  **Training Progress**: Log the loss and accuracy (or other relevant metrics) for each epoch.
5.  **Validation**: Log validation metrics at the end of each epoch or at specified intervals.
6.  **Final Evaluation**: Result of the evaluation on the test set (e.g., final accuracy, MAE, F1-score, confusion matrix).

The log file must be uploaded to `log/run.log` to the repository. The logs must be easy to understand and self explanatory. 
Ensure that `src/utils.py` is used to configure the logger so that output is directed to stdout (which Docker captures).

### Submission Checklist

Before submitting your project, ensure you have completed the following steps.
**Please note that the submission can only be accepted if these minimum requirements are met.**

- [ ] **Project Information**: Filled out the "Project Information" section (Topic, Name, Extra Credit).
- [ ] **Solution Description**: Provided a clear description of your solution, model, and methodology.
- [ ] **Extra Credit**: If aiming for +1 mark, filled out the justification section.
- [ ] **Data Preparation**: Included a script or precise description for data preparation.
- [ ] **Dependencies**: Updated `requirements.txt` with all necessary packages and specific versions.
- [ ] **Configuration**: Used `src/config.py` for hyperparameters and paths, contains at least the number of epochs configuration variable.
- [ ] **Logging**:
    - [ ] Log uploaded to `log/run.log`
    - [ ] Log contains: Hyperparameters, Data preparation and loading confirmation, Model architecture, Training metrics (loss/acc per epoch), Validation metrics, Final evaluation results, Inference results.
- [ ] **Docker**:
    - [ ] `Dockerfile` is adapted to your project needs.
    - [ ] Image builds successfully (`docker build -t dl-project .`).
    - [ ] Container runs successfully with data mounted (`docker run ...`).
    - [ ] The container executes the full pipeline (preprocessing, training, evaluation).
- [ ] **Cleanup**:
    - [ ] Removed unused files.
    - [ ] **Deleted this "Submission Instructions" section from the README.**

## Project Details

### Project Information

- **Selected Topic**: Legal Text Decoder
- **Student Name**: Kristóf Koleszár
- **Aiming for +1 Mark**: No

### Solution Description

### Problem Statement

This project focuses on automatically estimating the readability (difficulty level) of Hungarian legal texts using machine learning.
Legal documents are often hard to understand for non-experts, and the goal here is to classify how difficult a given text is to read on a **5-point scale**:

- **1 - Very Hard**: Extremely difficult to understand
- **2 - Hard**: Difficult to understand
- **3 - Moderate**: Somewhat understandable
- **4 - Easy**: Easy to understand
- **5 - Very Easy**: Very easy to understand

This is a **multi-class text classification** task using fine-tuned Hungarian BERT (HuBERT) model.

### Data Preparation

#### Raw Data Conversion (`convert_raw_data.py`)

The raw data comes from **Label Studio JSON annotation files** created by multiple student annotators.  
Each annotator worked in a separate folder.

The script `convert_raw_data.py` handles the conversion and filtering of the raw data:

- Automatically downloads the raw data archive from SharePoint
- Iterates through all student folders
- Excludes the following data:
  - `consensus/` (reserved for final evaluation)
  - `E77YIW/mak_aszf_cimkezes.json` (kept for inference demo)
- Maps Hungarian label names to numeric values (1–5)
- Removes duplicate texts (keeps the first occurrence)

The script produces three CSV files:

- **`neptun_data.csv`** (~3,300 samples) – main training dataset  
- **`evaluation.csv`** (132 samples) – consensus annotations for final evaluation  
- **`inference_demo.csv`** (105 samples) – unseen texts for inference demonstrations 

### Data Preprocessing (`01-data-preprocessing.py`)

The preprocessing pipeline prepares the data for model training:

1. **Text Cleaning**:
   - Whitespace normalization
   - Special character handling
   - Leading/trailing space removal

2. **Data Augmentation** (for regularization):
   - Word swap augmentation for minority classes
   - Word deletion augmentation (10% of words)
   - Applied only to classes with fewer than 250 samples
   - Augmentation ratio: 15%

3. **Stratified Split**:
   - Train: 70% (~2,398 samples)
   - Validation: 15% (~515 samples)
   - Test: 15% (~515 samples)
   - Stratification preserves class distribution across all splits

### Exploratory Data Analysis

Two Jupyter notebooks provide comprehensive data analysis:

#### Data Exploration (`01-data-exploration.ipynb`)
- Text length and word count distributions
- Text complexity metrics (sentence length, lexical diversity, word length)
- Word frequency analysis with Hungarian stopword removal
- Bigram and trigram analysis
- TF-IDF analysis by label class
- Dataset comparison (train vs evaluation)

#### Label Analysis (`02-label-analysis.ipynb`)
- Label distribution visualization
- Correlation analysis between complexity metrics and labels
- Linear vs polynomial fit analysis
- Outlier detection using Z-score and IQR methods
- Class imbalance analysis (imbalance ratio ~6:1)

### Model Training (`02-training.py`)

#### Baseline Model
- **Architecture**: TF-IDF vectorization + Logistic Regression
- **Features**: 5,000 max features, unigram + bigram
- **Regularization**: L2 penalty, class weight balancing
- **Purpose**: Provides a reference point for deep learning model comparison

#### HuBERT Model
- **Base Model**: `SZTAKI-HLT/hubert-base-cc` (Hungarian BERT)
- **Architecture**: Pre-trained HuBERT encoder + custom classification head
  - Dropout layers (0.3)
  - Batch normalization
  - Two-layer MLP classifier

#### Regularization Techniques
- **Dropout**: 0.3 probability in classification head
- **Weight Decay**: L2 regularization (0.01) via AdamW optimizer
- **Label Smoothing**: 0.1 for cross-entropy loss
- **Gradient Clipping**: Max norm 1.0
- **Early Stopping**: Patience of 3 epochs
- **Layer Freezing**: Only fine-tune last 4 encoder layers

#### Training Configuration
- **Optimizer**: AdamW with learning rate 2e-5
- **Scheduler**: Cosine annealing with warm restarts
- **Batch Size**: 16 (effective 32 with gradient accumulation)
- **Max Sequence Length**: 256 tokens
- **Mixed Precision**: FP16 training for GPU efficiency

### Evaluation (`03-evaluation.py`)

The evaluation script provides comprehensive model assessment:

#### Metrics
- **Accuracy**: Overall classification accuracy
- **F1-Score**: Weighted and macro-averaged
- **Precision/Recall**: Per-class and averaged
- **Confusion Matrix**: Visual representation of predictions

#### Outputs
- Model comparison table (Baseline vs HuBERT)
- Detailed classification reports
- Confusion matrix visualizations
- Per-class metric plots
- Prediction CSV with probabilities

### Inference (`04-inference.py`)

Supports three modes for running predictions on new data:
- **File mode**: Batch inference on CSV files
- **Interactive mode**: Real-time text classification
- **Text mode**: Single text prediction via command line

## Results

### Model Comparison

| Model | Validation Accuracy | Validation F1 (Weighted) |
|-------|---------------------|--------------------------|
| Baseline (TF-IDF + LogReg) | ~40% | ~37% |
| HuBERT (fine-tuned) | ~45% | ~43% |

### Training Progress

The HuBERT model was trained for 5-7 epochs before early stopping triggered:

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Val F1 |
|-------|------------|-----------|----------|---------|--------|
| 1 | 1.54 | 34.1% | 1.42 | 42.3% | 39.8% |
| 2 | 1.42 | 41.9% | 1.36 | 42.7% | 40.3% |
| 3 | 1.36 | 44.8% | 1.34 | 44.1% | 41.4% |
| 4 | 1.31 | 47.5% | 1.34 | 44.5% | 42.6% |
| 5 | 1.29 | 48.1% | 1.33 | 42.5% | 39.7% |

Best model saved at Epoch 4 with Val F1: 42.6%

### Key Findings

- **HuBERT outperforms baseline** by ~5% accuracy and ~6% F1-score
- **Class imbalance** affects performance on minority classes (Label 1, 2)
- **Early stopping** prevents overfitting after epoch 4-5
- **Regularization techniques** (dropout, weight decay, label smoothing) help generalization
- **Challenge**: 5-class classification with subjective labels is inherently difficult

### Docker Instructions

This project is containerized using Docker. Follow the instructions below to build and run the solution.

#### Build

Run the following command in the root directory of the repository to build the Docker image:

```bash
docker build -t my-dl-project-work-app:1.0 .
```

#### Run

To run the solution, use the following command. Log folder with the log file has to be created before running the script. It is default existing in the repo.

**In Windows PowerShell**:
```bash
docker run --rm --gpus all -v "$(pwd)/data:/data" -v "$(pwd)/output:/app/output" -v "$(pwd)/log:/log" my-dl-project-work-app:1.0 > log/run.log 2>&1
```

*   Replace `$(pwd)/data` with the actual path to your dataset on your host machine that meets the [Data preparation requirements](#data-preparation). It can be any folder, even unexisting, docker will create it for us automatically on host machine.
*   The `log/run.log 2>&1` part ensures that all output (standard output and errors) is saved to `log/run.log`.
*   The container is configured to run every step (data preprocessing, training, evaluation, inference).


### File Structure and Functions

[Update according to the final file structure.]

The repository is structured as follows:

- **`src/`**: Contains the source code for the machine learning pipeline.
    - `01-data-preprocessing.py`: Scripts for loading, cleaning, and preprocessing the raw data.
    - `02-training.py`: The main script for defining the model and executing the training loop.
    - `03-evaluation.py`: Scripts for evaluating the trained model on test data and generating metrics.
    - `04-inference.py`: Script for running the model on new, unseen data to generate predictions.
    - `config.py`: Configuration file containing hyperparameters (e.g., epochs) and paths.
    - `convert_raw_data.py`: The script extracts labeled texts and numeric labels from raw Label Studio JSON files, then exports them in CSV and JSON format split into three datasets (train/neptun, evaluation, inference demo).

- **`notebook/`**: Contains Jupyter notebooks for analysis and experimentation.
    - `01-data-exploration.ipynb`: Notebook for initial exploratory data analysis (EDA) and visualization.
    - `02-label-analysis.ipynb`: Notebook for analyzing the distribution and properties of the target labels.

- **`log/`**: Contains log files.
    - `run.log`: Example log file showing the output of a successful training run.

- **Root Directory**:
    - `Dockerfile`: Configuration file for building the Docker image with the necessary environment and dependencies.
    - `requirements.txt`: List of Python dependencies required for the project.
    - `README.md`: Project documentation and instructions.
    - `.gitignore`: Files not to be committed.