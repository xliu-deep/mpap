"""
Data loading utilities for MPAP model.
"""

import numpy as np
import torch
from typing import List, Tuple, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def load_tensor(file_path: str, dtype: torch.dtype, device: torch.device) -> List[torch.Tensor]:
    """
    Load tensor data from .npy file.
    
    Args:
        file_path: Path to .npy file (without extension)
        dtype: Tensor data type
        device: Device to load tensors on
        
    Returns:
        List of tensors
    """
    full_path = Path(file_path).with_suffix('.npy')
    if not full_path.exists():
        raise FileNotFoundError(f"File not found: {full_path}")
    
    data = np.load(str(full_path), allow_pickle=True)
    tensors = [dtype(d).to(device) for d in data]
    logger.debug(f"Loaded {len(tensors)} tensors from {full_path}")
    return tensors


def load_tensor_label(file_path: str, dtype: torch.dtype, device: torch.device) -> List[torch.Tensor]:
    """
    Load label tensor data from .npy file.
    
    Args:
        file_path: Path to .npy file (without extension)
        dtype: Tensor data type
        device: Device to load tensors on
        
    Returns:
        List of label tensors
    """
    full_path = Path(file_path).with_suffix('.npy')
    if not full_path.exists():
        raise FileNotFoundError(f"File not found: {full_path}")
    
    data = np.load(str(full_path), allow_pickle=True)
    data = data.reshape(data.shape[0], 1)
    tensors = [dtype(d).to(device) for d in data]
    logger.debug(f"Loaded {len(tensors)} label tensors from {full_path}")
    return tensors


def shuffle_dataset(dataset: List, seed: int = 1234) -> List:
    """
    Shuffle dataset with given seed.
    
    Args:
        dataset: Dataset to shuffle
        seed: Random seed
        
    Returns:
        Shuffled dataset
    """
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


def load_dataset(
    input_dir: str,
    device: torch.device,
    shuffle: bool = True,
    seed: int = 1234
) -> List[Tuple]:
    """
    Load complete dataset from input directory.
    
    Args:
        input_dir: Directory containing .npy files
        device: Device to load tensors on
        shuffle: Whether to shuffle the dataset
        seed: Random seed for shuffling
        
    Returns:
        List of tuples containing (fingerprints, pgraphs, padjacencies, graphs,
                                   morgans, adjacencies, sizes, labels, waters)
    """
    input_path = Path(input_dir)
    
    logger.info(f"Loading dataset from {input_dir}")
    
    fingerprints = load_tensor(str(input_path / 'fingerprints'), torch.FloatTensor, device)
    pgraphs = load_tensor(str(input_path / 'pgraph'), torch.FloatTensor, device)
    padjacencies = load_tensor(str(input_path / 'padjs'), torch.FloatTensor, device)
    graphs = load_tensor(str(input_path / 'graph'), torch.FloatTensor, device)
    morgans = load_tensor(str(input_path / 'morgan'), torch.LongTensor, device)
    adjacencies = load_tensor(str(input_path / 'adjacencies'), torch.FloatTensor, device)
    sizes = load_tensor_label(str(input_path / 'size'), torch.FloatTensor, device)
    labels = load_tensor_label(str(input_path / 'label'), torch.FloatTensor, device)
    waters = load_tensor_label(str(input_path / 'water'), torch.FloatTensor, device)
    
    dataset = list(zip(
        fingerprints, pgraphs, padjacencies, graphs,
        morgans, adjacencies, sizes, labels, waters
    ))
    
    if shuffle:
        dataset = shuffle_dataset(dataset, seed)
    
    logger.info(f"Loaded {len(dataset)} samples")
    return dataset

