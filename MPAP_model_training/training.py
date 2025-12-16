"""
Refactored training script using the new configuration and utility system.
"""

import sys
import os
from pathlib import Path

# Get the project root directory (parent of MPAP_model_training)
_project_root = Path(__file__).parent.parent.resolve()

# Add project root to path to import mpap package
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Change to project root directory so relative paths in config work correctly
os.chdir(_project_root)

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import R2Score

# Import new utilities
from mpap.config import Config
from mpap.utils import setup_logging, get_device, setup_seed, create_output_dir
from mpap.data_loader import load_dataset

# Import model components from model.py in the same directory
import importlib.util
model_path = Path(__file__).parent / "model.py"
if not model_path.exists():
    raise FileNotFoundError(f"Model file not found: {model_path}")

spec = importlib.util.spec_from_file_location("model", model_path)
model_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_module)

Predictor = model_module.Predictor
ResMLP = model_module.ResMLP
Affine = model_module.Affine
pack = model_module.pack
loss_fn = torch.nn.MSELoss()
mae_loss_fn = torch.nn.L1Loss()
smoothl1_loss_fn = torch.nn.SmoothL1Loss()

import logging
logger = logging.getLogger(__name__)


def get_metrics(model, dataset_test, batch_size, device):
    """
    Evaluate model on test dataset.
    
    Args:
        model: Trained model
        dataset_test: Test dataset
        batch_size: Batch size for evaluation
        device: Device to run on
        
    Returns:
        Tuple of (loss, mae_loss, r2, outputs, labels)
    """
    model.eval()
    valid_outputs = []
    valid_labels = []
    valid_loss = []
    valid_mae_loss = []
    
    r2_score = R2Score().to(device)
    
    N = len(dataset_test)
    i = 0
    fingerprints, pgraphs, padjacencys, graphs, morgans, adjs, sizes, labels, waters = (
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
            labels.append(label)
            waters.append(water)
            
            if i % batch_size == 0 or i == N:
                # Pack batch
                packed = pack(
                    fingerprints, pgraphs, padjacencys, graphs,
                    morgans, adjs, sizes, labels, waters, device
                )
                fingerprints1, pgraphs1, padjacencys1, graphs1, morgans1, adjs1, sizes1, labels1, waters1 = packed
                data_batch = (fingerprints1, pgraphs1, padjacencys1, graphs1, morgans1, adjs1, sizes1, labels1, waters1)
                
                # Forward pass
                outputs = model(data_batch)
                
                # Calculate losses
                labels_tensor = torch.tensor(labels, device=device).float()
                loss = loss_fn(outputs, labels_tensor)
                mae_loss = mae_loss_fn(outputs, labels_tensor)
                
                # Store results
                outputs_list = outputs.detach().cpu().numpy().tolist()
                valid_outputs.extend(outputs_list)
                valid_loss.append(loss.cpu().detach().numpy())
                valid_mae_loss.append(mae_loss.cpu().detach().numpy())
                valid_labels.extend(labels)
                
                # Reset batch
                fingerprints, pgraphs, padjacencys, graphs, morgans, adjs, sizes, labels, waters = (
                    [], [], [], [], [], [], [], [], []
                )
    
    # Calculate metrics
    avg_loss = np.mean(np.array(valid_loss).flatten())
    avg_mae_loss = np.mean(np.array(valid_mae_loss).flatten())
    r2_value = r2_score(
        torch.tensor(valid_outputs, device=device).float(),
        torch.tensor(valid_labels, device=device).float()
    )
    # Extract scalar value from tensor
    r2 = r2_value.item() if isinstance(r2_value, torch.Tensor) else float(r2_value)
    
    return avg_loss, avg_mae_loss, r2, valid_outputs, valid_labels


def train_epoch(model, dataset_train, batch_size, optimizer, device):
    """
    Train for one epoch.
    
    Args:
        model: Model to train
        dataset_train: Training dataset
        batch_size: Batch size
        optimizer: Optimizer
        device: Device to run on
        
    Returns:
        Tuple of (avg_loss, avg_mae_loss)
    """
    model.train()
    np.random.shuffle(dataset_train)
    
    running_loss = []
    running_mae_loss = []
    
    N = len(dataset_train)
    i = 0
    fingerprints, pgraphs, padjacencys, graphs, morgans, adjs, sizes, labels, waters = (
        [], [], [], [], [], [], [], [], []
    )
    
    for data in dataset_train:
        i += 1
        fingerprint, pgraph, padjacency, graph, morgan, adj, size, label, water = data
        fingerprints.append(fingerprint)
        pgraphs.append(pgraph)
        padjacencys.append(padjacency)
        graphs.append(graph)
        morgans.append(morgan)
        adjs.append(adj)
        sizes.append(size)
        labels.append(label)
        waters.append(water)
        
        if i % batch_size == 0 or i == N:
            optimizer.zero_grad()
            
            # Pack batch
            packed = pack(
                fingerprints, pgraphs, padjacencys, graphs,
                morgans, adjs, sizes, labels, waters, device
            )
            fingerprints1, pgraphs1, padjacencys1, graphs1, morgans1, adjs1, sizes1, labels1, waters1 = packed
            data_batch = (fingerprints1, pgraphs1, padjacencys1, graphs1, morgans1, adjs1, sizes1, labels1, waters1)
            
            # Forward pass
            outputs = model(data_batch)
            
            # Calculate loss
            labels_tensor = torch.tensor(labels, device=device).float()
            loss = smoothl1_loss_fn(outputs, labels_tensor)
            mae_loss = mae_loss_fn(outputs, labels_tensor)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Store metrics
            running_loss.append(loss.cpu().detach().item())
            running_mae_loss.append(mae_loss.cpu().detach().item())
            
            # Reset batch
            fingerprints, pgraphs, padjacencys, graphs, morgans, adjs, sizes, labels, waters = (
                [], [], [], [], [], [], [], [], []
            )
    
    avg_loss = np.mean(running_loss) if running_loss else 0.0
    avg_mae_loss = np.mean(running_mae_loss) if running_mae_loss else 0.0
    
    return avg_loss, avg_mae_loss


def train(config: Config):
    """
    Main training function.
    
    Args:
        config: Configuration object
    """
    # Setup
    setup_seed(config.get('training.seed'))
    device = get_device(config.get('device.cuda_device'))
    
    # Set number of threads
    if config.get('device.num_threads'):
        torch.set_num_threads(config.get('device.num_threads'))
    
    logger.info("Starting training...")
    logger.info(f"Device: {device}")
    logger.info(f"Configuration: {config.config_path}")
    
    # Load datasets
    logger.info("Loading datasets...")
    train_dir = config.get('paths.train_input_dir')
    valid_dir = config.get('paths.valid_input_dir')
    
    dataset_train = load_dataset(train_dir, device, shuffle=True, seed=config.get('training.seed'))
    dataset_test = load_dataset(valid_dir, device, shuffle=True, seed=config.get('training.seed'))
    
    logger.info(f"Training samples: {len(dataset_train)}")
    logger.info(f"Validation samples: {len(dataset_test)}")
    
    # Initialize model
    logger.info("Initializing model...")
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
    
    # Initialize optimizer and scheduler
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.get('training.learning_rate'),
        weight_decay=config.get('training.weight_decay')
    )
    scheduler = ReduceLROnPlateau(optimizer, patience=5, mode='min', verbose=True)
    
    # Training loop
    best_val_loss = float('inf')
    consecutive_no_improve = 0
    patience = config.get('training.early_stopping_patience', 15)
    
    # Create output directories
    model_dir = create_output_dir(config.get('paths.model_dir'))
    output_dir = create_output_dir(config.get('paths.output_dir'))
    log_file = os.path.join(output_dir, 'training_log.txt')
    
    logger.info(f"Model directory: {model_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    max_epochs = config.get('training.max_epochs')
    batch_size = config.get('training.batch_size')
    
    for epoch in range(max_epochs):
        logger.info(f"Epoch {epoch + 1}/{max_epochs}")
        
        # Train
        train_loss, train_mae = train_epoch(model, dataset_train, batch_size, optimizer, device)
        
        # Validate
        val_loss, val_mae, r2, outputs, labels = get_metrics(model, dataset_test, batch_size, device)
        scheduler.step(val_loss)
        
        # Log metrics
        log_msg = (
            f"Epoch {epoch + 1}: "
            f"train_loss={train_loss:.6f}, train_mae={train_mae:.6f}, "
            f"val_loss={val_loss:.6f}, val_mae={val_mae:.6f}, val_r2={r2:.6f}"
        )
        logger.info(log_msg)
        
        # Save training log
        with open(log_file, 'a') as f:
            f.write(f"{epoch + 1}\t{train_loss:.6f}\t{train_mae:.6f}\t{val_loss:.6f}\t{val_mae:.6f}\t{r2:.6f}\n")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            consecutive_no_improve = 0
            
            model_path = os.path.join(model_dir, f"{best_val_loss:.6f}.tar")
            torch.save(model.state_dict(), model_path)
            logger.info(f"Saved best model: {model_path}")
        else:
            consecutive_no_improve += 1
        
        # Early stopping
        if consecutive_no_improve >= patience:
            logger.info(f"Early stopping after {epoch + 1} epochs (no improvement for {patience} epochs)")
            break
    
    logger.info("Training completed!")
    logger.info(f"Best validation loss: {best_val_loss:.6f}")


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
    
    # Run training
    try:
        train(config)
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
