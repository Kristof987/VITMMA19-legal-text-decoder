import logging
import sys
import torch
import numpy as np
import random


def setup_logger(name="legal_text_classifier", level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    logger.handlers = []
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    
    return logger


def get_logger(name="legal_text_classifier"):
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
