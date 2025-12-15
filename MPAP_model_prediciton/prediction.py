"""
Refactored prediction script using the new configuration and utility system.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path to import mpap package
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import pandas as pd
from torchmetrics import R2Score

# Import new utilities
from mpap.config import Config
from mpap.utils import setup_logging, get_device, setup_seed, create_output_dir
from mpap.data_loader import load_dataset

# Import model components from model.py
import importlib.util
model_path = Path(__file__).parent.parent / "MPAP_model_training" / "model.py"
if not model_path.exists():
    raise FileNotFoundError(f"Model file not found: {model_path}")

spec = importlib.util.spec_from_file_location("model", model_path)
model_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_module)

Predictor = model_module.Predictor
ResMLP = model_module.ResMLP
Affine = model_module.Affine
pack = model_module.pack

import logging
logger = logging.getLogger(__name__)


def predict(model, dataset_test, batch_size, device):
    """
    Make predictions on test dataset.
    
    Args:
        model: Trained model
        dataset_test: Test dataset
        batch_size: Batch size for prediction
        device: Device to run on
        
    Returns:
        Tuple of (predictions, labels, metrics_dict)
    """
    model.eval()
    predictions = []
    labels = []
    losses = []
    mae_losses = []
    
    loss_fn = torch.nn.MSELoss()
    mae_loss_fn = torch.nn.L1Loss()
    r2_score = R2Score().to(device)
    
    N = len(dataset_test)
    i = 0
    fingerprints, pgraphs, padjacencys, graphs, morgans, adjs, sizes, label_list, waters = (
        [], [], [], [], [], [], [], [], []
    )
    
    with torch.no_grad():
        for data in dataset_test:
            i += 1
            fingerprint, pgraph, padjacency, graph, morgan, adj, size, label, water = data
            fingerprints.append(fingerprint)
            pgraphs.append(pgraph)
            padjacencys.append(padjacency)
            graphs.append(graph)
            morgans.append(morgan)
            adjs.append(adj)
            sizes.append(size)
            label_list.append(label)
            waters.append(water)
            
            if i % batch_size == 0 or i == N:
                # Pack batch
                packed = pack(
                    fingerprints, pgraphs, padjacencys, graphs,
                    morgans, adjs, sizes, label_list, waters, device
                )
                fingerprints1, pgraphs1, padjacencys1, graphs1, morgans1, adjs1, sizes1, labels1, waters1 = packed
                data_batch = (fingerprints1, pgraphs1, padjacencys1, graphs1, morgans1, adjs1, sizes1, labels1, waters1)
                
                # Forward pass
                outputs = model(data_batch)
                
                # Calculate metrics
                labels_tensor = torch.tensor(label_list, device=device).float()
                loss = loss_fn(outputs, labels_tensor)
                mae_loss = mae_loss_fn(outputs, labels_tensor)
                
                # Store results
                outputs_list = outputs.detach().cpu().numpy().tolist()
                predictions.extend(outputs_list)
                labels.extend(label_list)
                losses.append(loss.cpu().detach().numpy())
                mae_losses.append(mae_loss.cpu().detach().numpy())
                
                # Reset batch
                fingerprints, pgraphs, padjacencys, graphs, morgans, adjs, sizes, label_list, waters = (
                    [], [], [], [], [], [], [], [], []
                )
    
    # Calculate final metrics
    avg_loss = np.mean(np.array(losses).flatten())
    avg_mae = np.mean(np.array(mae_losses).flatten())
    r2 = r2_score(
        torch.tensor(predictions, device=device).float(),
        torch.tensor(labels, device=device).float()
    )
    
    metrics = {
        'mse': float(avg_loss),
        'mae': float(avg_mae),
        'r2': float(r2)
    }
    
    return predictions, labels, metrics


def load_model(model_path: str, config: Config, device: torch.device):
    """
    Load trained model from checkpoint.
    
    Args:
        model_path: Path to model checkpoint
        config: Configuration object
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    logger.info(f"Loading model from {model_path}")
    
    model = Predictor(
        ResMLP, Affine,
        dim=config.get('model.dim'),
        window=config.get('model.window'),
        window2=config.get('model.window2'),
        layer_gnn=config.get('model.layer_gnn'),
        layer_cnn=config.get('model.layer_cnn'),
        layer_output=config.get('model.layer_output'),
        dropout=config.get('model.dropout')
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    
    logger.info("Model loaded successfully")
    return model


def main():
    """Main entry point."""
    # Load configuration
    config_path = os.getenv('MPAP_CONFIG', None)
    config = Config(config_path)
    
    # Setup logging
    setup_logging(
        log_dir=config.get('logging.log_dir'),
        log_file=config.get('logging.log_file'),
        level=config.get('logging.level')
    )
    
    logger.info("Starting prediction...")
    
    # Setup device
    setup_seed(config.get('training.seed'))
    device = get_device(config.get('device.cuda_device'))
    logger.info(f"Using device: {device}")
    
    # Load model
    # Default to best-model if not specified
    model_dir = config.get('paths.model_dir')
    model_file = os.path.join(model_dir, 'best-model', '0.47535303.tar')
    
    # Allow override via environment variable
    model_path = os.getenv('MPAP_MODEL_PATH', model_file)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = load_model(model_path, config, device)
    
    # Load test dataset
    test_dir = config.get('paths.test_input_dir')
    logger.info(f"Loading test dataset from {test_dir}")
    dataset_test = load_dataset(test_dir, device, shuffle=False)
    logger.info(f"Test samples: {len(dataset_test)}")
    
    # Make predictions
    batch_size = config.get('training.batch_size')
    predictions, labels, metrics = predict(model, dataset_test, batch_size, device)
    
    # Log metrics
    logger.info("Prediction Metrics:")
    logger.info(f"  MSE: {metrics['mse']:.6f}")
    logger.info(f"  MAE: {metrics['mae']:.6f}")
    logger.info(f"  R²:  {metrics['r2']:.6f}")
    
    # Save predictions
    output_dir = create_output_dir(config.get('paths.output_dir'))
    output_file = os.path.join(output_dir, 'predictions.csv')
    
    df = pd.DataFrame({
        'labels': labels,
        'predictions': predictions
    })
    df.to_csv(output_file, index=False, sep='\t')
    logger.info(f"Predictions saved to {output_file}")
    
    # Save metrics
    metrics_file = os.path.join(output_dir, 'metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write(f"MSE: {metrics['mse']:.6f}\n")
        f.write(f"MAE: {metrics['mae']:.6f}\n")
        f.write(f"R²:  {metrics['r2']:.6f}\n")
    logger.info(f"Metrics saved to {metrics_file}")


if __name__ == "__main__":
    main()

