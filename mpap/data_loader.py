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
    
    # Handle object arrays (e.g., pgraph, padjs, graph, adjacencies)
    # Each element in object array is itself a numpy array
    if data.dtype == np.object_:
        tensors = []
        for d in data:
            # Handle nested object arrays (recursively convert to proper numpy array)
            arr = np.array(d, copy=False)
            while arr.dtype == np.object_ and len(arr) > 0:
                # If still object type, try to convert first element
                # This handles nested object arrays
                if isinstance(arr[0], np.ndarray):
                    # All elements should be arrays, try to stack them
                    try:
                        arr = np.vstack([np.array(item, copy=False) for item in arr])
                        break
                    except (ValueError, TypeError):
                        # If can't stack (different shapes), take first element
                        arr = np.array(arr[0], copy=False)
                else:
                    arr = np.array(arr[0], copy=False)
            
            # Convert to appropriate dtype for tensor conversion
            if dtype == torch.FloatTensor:
                # Convert any numeric type to float32 (including int types)
                if arr.dtype != np.float32:
                    arr = arr.astype(np.float32)
            elif dtype == torch.LongTensor:
                # Convert any integer type to int64
                if arr.dtype != np.int64:
                    arr = arr.astype(np.int64)
            
            tensors.append(dtype(torch.from_numpy(arr)).to(device))
    else:
        # Handle regular arrays (e.g., fingerprints, morgan)
        # For 2D arrays, iterate over first dimension
        if len(data.shape) > 1:
            tensors = []
            for d in data:
                arr = np.array(d, copy=False)
                # Convert to appropriate dtype
                if dtype == torch.FloatTensor and arr.dtype == np.float64:
                    arr = arr.astype(np.float32)
                elif dtype == torch.LongTensor and arr.dtype != np.int64:
                    arr = arr.astype(np.int64)
                tensors.append(dtype(torch.from_numpy(arr)).to(device))
        else:
            # 1D array - iterate over each element (scalar values)
            arr = np.array(data, copy=False)
            if dtype == torch.FloatTensor and arr.dtype == np.float64:
                arr = arr.astype(np.float32)
            elif dtype == torch.LongTensor and arr.dtype != np.int64:
                arr = arr.astype(np.int64)
            tensors = [dtype(torch.from_numpy(np.array([d]))).to(device) for d in arr]
    
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
    # Reshape to (n_samples, 1) for scalar labels
    if len(data.shape) == 1:
        data = data.reshape(data.shape[0], 1)
    
    # Convert dtype if needed (before iterating)
    arr = np.array(data, copy=False)
    if dtype == torch.FloatTensor:
        # Convert any numeric type to float32 (including int types like int64)
        if arr.dtype not in (np.float32, np.float16):
            arr = arr.astype(np.float32)
    elif dtype == torch.LongTensor:
        # Convert any integer type to int64
        if arr.dtype != np.int64:
            arr = arr.astype(np.int64)
    
    # Now iterate and convert to tensors
    # Each d is a 1D array (shape: (1,)), convert it properly
    tensors = []
    for d in arr:
        # Ensure d is float32 for FloatTensor
        if dtype == torch.FloatTensor and d.dtype != np.float32:
            d = d.astype(np.float32)
        elif dtype == torch.LongTensor and d.dtype != np.int64:
            d = d.astype(np.int64)
        tensors.append(dtype(torch.from_numpy(d)).to(device))
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

