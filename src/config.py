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
MODEL_NAME = "SZTAKI-HLT/hubert-base-cc"
NUM_CLASSES = 5
MAX_LENGTH = 256
HIDDEN_SIZE = 768
CLASSIFIER_HIDDEN_SIZE = 384
TEST_SIZE = 0.15
VAL_SIZE = 0.15

# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================
EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
ACCUMULATION_STEPS = 2

# ============================================================================
# REGULARIZATION
# ============================================================================
DROPOUT_RATE = 0.3
EARLY_STOPPING_PATIENCE = 3
GRADIENT_CLIP_VALUE = 1.0
LABEL_SMOOTHING = 0.1

# ============================================================================
# OPTIMIZATION
# ============================================================================
USE_MIXED_PRECISION = True

# ============================================================================
# REPRODUCIBILITY
# ============================================================================
RANDOM_SEED = 42

# ============================================================================
# LOGGING
# ============================================================================
LOG_FILE = "log/run.log"
LOG_INTERVAL = 10
SAVE_BEST_ONLY = True

# ============================================================================
# BASELINE MODEL
# ============================================================================
BASELINE_MAX_FEATURES = 5000
BASELINE_NGRAM_RANGE = (1, 2)

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
