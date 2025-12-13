"""
Configuration file for training hyperparameters and paths.
All configurable parameters are centralized here.
"""

from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================
DATA_DIR = Path("/data/processed")
LOG_DIR = Path("/log")
MODEL_DIR = Path("/data/output/models")
CHECKPOINT_DIR = Path("/data/output/checkpoints")
TENSORBOARD_DIR = Path("/data/output/tensorboard")

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
# Pre-trained Hungarian BERT model
MODEL_NAME = "SZTAKI-HLT/hubert-base-cc"
NUM_CLASSES = 5
MAX_LENGTH = 128  # Reduced from 256 to save memory
HIDDEN_SIZE = 768  # BERT hidden size
CLASSIFIER_HIDDEN_SIZE = 256  # Classification head hidden layer

# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================
EPOCHS = 5  # Reduced for faster CPU training
BATCH_SIZE = 4  # Further reduced for CPU training
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01  # L2 regularization (AdamW)
WARMUP_RATIO = 0.1  # 10% of training steps for warmup
ACCUMULATION_STEPS = 4  # Gradient accumulation to simulate batch_size=16

# ============================================================================
# REGULARIZATION
# ============================================================================
DROPOUT_RATE = 0.3  # Dropout probability
EARLY_STOPPING_PATIENCE = 3  # Stop if no improvement for N epochs
GRADIENT_CLIP_VALUE = 1.0  # Gradient clipping threshold
LABEL_SMOOTHING = 0.1  # Label smoothing for cross-entropy

# ============================================================================
# OPTIMIZATION
# ============================================================================
USE_MIXED_PRECISION = True  # Enable mixed precision training (float16)

# ============================================================================
# REPRODUCIBILITY
# ============================================================================
RANDOM_SEED = 42

# ============================================================================
# LOGGING
# ============================================================================
LOG_FILE = "log/run.log"
LOG_INTERVAL = 10  # Log every N batches
SAVE_BEST_ONLY = True  # Only save best model checkpoint

# ============================================================================
# BASELINE MODEL
# ============================================================================
BASELINE_MAX_FEATURES = 5000  # TF-IDF max features
BASELINE_NGRAM_RANGE = (1, 2)  # Unigrams and bigrams

# ============================================================================
# LABEL NAMES
# ============================================================================
LABEL_NAMES = {
    1: 'Very Hard',
    2: 'Hard',
    3: 'Moderate',
    4: 'Easy',
    5: 'Very Easy'
}
