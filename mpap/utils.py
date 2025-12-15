"""
Utility functions for MPAP model.
"""

import os
import random
import numpy as np
import torch
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def setup_seed(seed: int = 1234):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")


def get_device(cuda_device: Optional[int] = None) -> torch.device:
    """
    Get the appropriate device (CPU or CUDA).
    
    Args:
        cuda_device: Specific CUDA device ID, or None for auto-detection
        
    Returns:
        torch.device object
    """
    if cuda_device is not None:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{cuda_device}")
            logger.info(f"Using CUDA device {cuda_device}")
        else:
            device = torch.device("cpu")
            logger.warning(f"CUDA device {cuda_device} requested but not available. Using CPU.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA (auto-detected)")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    
    return device


def setup_logging(log_dir: str = "./logs", log_file: str = "mpap.log", level: str = "INFO"):
    """
    Setup logging configuration.
    
    Args:
        log_dir: Directory for log files
        log_file: Log file name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    logger.info(f"Logging initialized. Log file: {log_path}")


def create_output_dir(output_dir: str) -> str:
    """
    Create output directory if it doesn't exist.
    
    Args:
        output_dir: Path to output directory
        
    Returns:
        Absolute path to output directory
    """
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    return output_dir

