# MPAP: Microplastic-Pollutant Adsorption Prediction Model

A deep learning model for predicting adsorption coefficients (Kd) between microplastics and organic pollutants using multimodal representations.

## Overview

This project implements a multimodal Siamese neural network (MPAP) that predicts the adsorption coefficient (Kd) between microplastics and organic pollutants. The model integrates:

- **Molecular fingerprints**: ECFPs for pollutants and PolyBERT for microplastics
- **Graph-based structural features**: Extracted from atomic/bond descriptors
- **Environmental parameters**: MP particle size and water environment type



## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- Conda or pip

### Setup

1. **Clone the repository:**
```bash
git clone <repository-url>
cd github
```

2. **Create a virtual environment:**
```bash
# Using conda (recommended)
conda env create -f environment.yaml
conda activate kd

# Or using pip
pip install -r requirements.txt
```

3. **Install the package in development mode:**
```bash
pip install -e .
```

## Quick Start

### Step 1: Configure the Project

Edit `config/config.yaml` to set your data paths and hyperparameters:

```yaml
paths:
  train_input_dir: "./MPAP_model_training/train_input"
  valid_input_dir: "./MPAP_model_training/valid_input"
  test_input_dir: "./MPAP_model_training/test_input"
  model_dir: "./MPAP_model_training/models"
  output_dir: "./outputs"
  predata_input: "./MPAP_dataset/test.txt"
  predata_output: "./MPAP_predata/output"
```

### Step 2: Preprocess Data

Convert text data to numpy input files:

```bash
python MPAP_predata/predata.py
```

Or with custom input/output paths:

```bash
python MPAP_predata/predata.py --input your_data.txt --output output_directory/
```

**Input file format**: Tab-separated text file with columns:
- `category`: Type of microplastic (examples: PE, PS, PVC, PP, PET, PA)
- `psmiles`: SMILES of microplastic (examples: `[*]C=C[*]`, `[*]C=C[*]c1ccccc1`, `[*]C=C[*]Cl`, `[*]CC([*])C`, `[*]OCCOC(=O)c1ccc(C([*])=O)cc1`, `[*]NCCCCCCCC(=O)[*]`)
- `compound`: Name of pollutant (examples: `biphenyl (BIP)`, `Atrazine`)
- `smiles`: SMILES of pollutant (examples: `C1=CC=C(C=C1)C2=CC=CC=C2`, `CCNC1=NC(=NC(=N1)Cl)NC(C)C`)
- `average size`: Particle size in micrometers (examples: `150`, `261`)
- `water3`: Water type (1=freshwater, 2=ultrapure, 3=seawater)
- `logkd`: Actual Kd value 
- `poly_smiles`: SMILES of polymer

### Step 3: Train the Model

Train the model with default configuration:

```bash
python MPAP_model_training/training.py
```

The script will:
- Load configuration from `config/config.yaml`
- Load training and validation datasets
- Train the model with specified hyperparameters
- Save best models to `MPAP_model_training/models/`
- Log training progress to `logs/mpap.log`
- Save training metrics to `outputs/training_log.txt`

### Step 4: Make Predictions

Make predictions using a trained model:

```bash
python MPAP_model_prediciton/prediction.py
```

Or specify a custom model path:

```bash
MPAP_MODEL_PATH=./path/to/model.tar python MPAP_model_prediciton/prediction.py
```

Predictions will be saved to:
- `outputs/predictions.csv`: Predictions and labels
- `outputs/metrics.txt`: Performance metrics (MSE, MAE, R²)

## Configuration

All hyperparameters and paths are configured in `config/config.yaml`. Key settings include:

### Model Parameters
```yaml
model:
  dim: 75                    # Feature dimension
  layer_gnn: 3               # Number of GNN layers
  layer_cnn: 3               # Number of CNN layers
  dropout: 0.012             # Dropout rate
```

### Training Parameters
```yaml
training:
  max_epochs: 200            # Maximum training epochs
  batch_size: 128            # Batch size
  learning_rate: 0.0007      # Learning rate
  early_stopping_patience: 15 # Early stopping patience
  seed: 1234                 # Random seed
```

### Device Configuration
```yaml
device:
  cuda_device: null          # null for auto, or specify device ID
  num_threads: 5             # Number of CPU threads
```

### Paths
```yaml
paths:
  train_input_dir: "./MPAP_model_training/train_input"
  valid_input_dir: "./MPAP_model_training/valid_input"
  test_input_dir: "./MPAP_model_training/test_input"
  model_dir: "./MPAP_model_training/models"
  output_dir: "./outputs"
```

## Environment Variables

You can override configuration settings using environment variables:

```bash
# Use custom config file
export MPAP_CONFIG=config/custom_config.yaml

# Specify model path for prediction
export MPAP_MODEL_PATH=./models/best_model.tar

# Then run scripts normally
python MPAP_model_training/training.py
```

## Project Structure

```
.
├── mpap/                      # Main package
│   ├── __init__.py
│   ├── config.py             # Configuration management
│   ├── utils.py              # Utility functions (logging, device, etc.)
│   └── data_loader.py        # Data loading utilities
├── config/                   # Configuration files
│   └── config.yaml           # Main configuration file
├── MPAP_dataset/             # Dataset files
│   ├── train.txt
│   ├── valid.txt
│   └── test.txt
├── MPAP_model_training/      # Training scripts
│   ├── training.py           # Main training script
│   ├── model.py              # Model architecture
│   ├── train_input/          # Training data (numpy files)
│   └── valid_input/          # Validation data (numpy files)
├── MPAP_model_prediction/    # Prediction scripts
│   ├── prediction.py        # Main prediction script
│   └── test_input/           # Test data (numpy files)
├── MPAP_predata/             # Data preprocessing
│   ├── predata.py           # Main preprocessing script
│   ├── graph_features.py    # Graph feature extraction
│   └── prints.py            # Fingerprint generation
├── requirements.txt          # Python dependencies
├── environment.yaml          # Conda environment
└── README.md                 # This file
```

## Usage Examples

### Using the Python API

```python
from mpap.config import Config
from mpap.utils import setup_logging, get_device
from mpap.data_loader import load_dataset

# Load configuration
config = Config()

# Setup logging
setup_logging(
    log_dir=config.get('logging.log_dir'),
    level=config.get('logging.level')
)

# Get device
device = get_device(config.get('device.cuda_device'))

# Load dataset
train_data = load_dataset(
    config.get('paths.train_input_dir'),
    device=device
)
```

### Custom Configuration

Create a custom config file:

```yaml
# custom_config.yaml
model:
  dim: 100
  dropout: 0.05

training:
  batch_size: 64
  learning_rate: 0.001
```

Then use it:

```bash
MPAP_CONFIG=custom_config.yaml python MPAP_model_training/training.py
```

## Data Format

### Input Text File

Tab-separated file with header:
```
category	psmiles	compound	smiles	average size	water3	logkd	poly_smiles
PE	[*]C=C[*]	Benzene	C1=CC=CC=C1	100	1	2.5	[*]C=C[*]
```

### Output Numpy Files

After preprocessing, the following `.npy` files are generated:
- `fingerprints.npy`: Microplastic fingerprints (600-dim)
- `pgraph.npy`: Microplastic graph features
- `padjs.npy`: Microplastic adjacency matrices
- `graph.npy`: Pollutant graph features
- `adjacencies.npy`: Pollutant adjacency matrices
- `morgan.npy`: Morgan fingerprints (2048-dim)
- `size.npy`: Particle sizes
- `water.npy`: Water type (1, 2, or 3)
- `label.npy`: Target Kd values

## Model Architecture

The MPAP model consists of:

1. **Graph Neural Networks (GNN)**: Process molecular graphs for both pollutants and microplastics
2. **Fingerprint Encoders**: Process ECFP and PolyBERT fingerprints
3. **Attention Mechanisms**: Multi-head attention for feature fusion
4. **Fusion Layers**: Combine multimodal representations
5. **Output Layers**: Predict log10(Kd) values

## Output Files

### Training
- **Models**: `MPAP_model_training/models/best_model_*.tar`
- **Logs**: `logs/mpap.log`
- **Metrics**: `outputs/training_log.txt` (epoch, train_loss, val_loss, val_mae, val_r2)

### Prediction
- **Predictions**: `outputs/predictions.csv` (labels and predictions)
- **Metrics**: `outputs/metrics.txt` (MSE, MAE, R²)

## Troubleshooting

### Import Errors
If you get import errors:
```bash
pip install -e .
```

### Config Not Found
Default config is at `config/config.yaml`. Override with:
```bash
MPAP_CONFIG=/path/to/config.yaml python script.py
```

### Model Not Found
For prediction, set model path:
```bash
MPAP_MODEL_PATH=./MPAP_model_prediction/best-model/0.47535303.tar python MPAP_model_prediction/prediction.py
```

### CUDA Out of Memory
Reduce batch size in `config/config.yaml`:
```yaml
training:
  batch_size: 64  # Reduce from 128
```

### Data Not Found
Check that data directories exist and contain `.npy` files:
```bash
ls MPAP_model_training/train_input/
# Should show: fingerprints.npy, graph.npy, etc.
```

## Performance

Model performance metrics:
- **R² Score**: Measures explained variance
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error

## Workflow

### Step 1: Convert text to npy input file using MPAP predata

Preprocess your data from text format to numpy arrays:

```bash
python MPAP_predata/predata.py --input your_data.txt --output output_directory/
```

### Step 2: Train the model

Train the model with your preprocessed data:

```bash
python MPAP_model_training/training.py
```

### Step 3: Predict Kd values

Use the trained model to make predictions:

```bash
python MPAP_model_prediciton/prediction.py
```

## Contact

For questions or issues, please open an issue on GitHub or contact csxyllx@gmail.com.





