# Migration Guide: Using Refactored Scripts

This guide explains how to use the newly refactored scripts that integrate with the configuration and utility system.

## Overview

Three refactored scripts have been created:
1. **`training_refactored.py`** - Training script using config system
2. **`prediction_refactored.py`** - Prediction script using config system  
3. **`predata_refactored.py`** - Data preprocessing script using config system

## Key Improvements

### Before (Original Scripts)
- Hardcoded paths: `'D:/microplastics/model/polyDTA/train_input/'`
- Hardcoded hyperparameters in code
- No logging system
- Manual device management
- No error handling

### After (Refactored Scripts)
- ✅ Configuration-based paths from `config.yaml`
- ✅ All hyperparameters in config file
- ✅ Comprehensive logging to files and console
- ✅ Automatic device detection/management
- ✅ Proper error handling and exceptions

## Usage

### 1. Data Preprocessing

**Original:**
```bash
# Had to modify line 889 in predata.py
python MPAP_predata/predata.py
```

**Refactored:**
```bash
# Uses config.yaml or command-line arguments
python MPAP_predata/predata_refactored.py

# Or with custom input/output
python MPAP_predata/predata_refactored.py --input data.txt --output output_dir/
```

### 2. Training

**Original:**
```bash
# Hardcoded paths and parameters
python MPAP_model_training/training.py
```

**Refactored:**
```bash
# Uses config.yaml for all settings
python MPAP_model_training/training_refactored.py

# Or with custom config
MPAP_CONFIG=config/custom_config.yaml python MPAP_model_training/training_refactored.py
```

### 3. Prediction

**Original:**
```bash
# Had to modify lines 74, 781, 176, 177, 934 in predication.py
python MPAP_model_prediciton/predication.py
```

**Refactored:**
```bash
# Uses config.yaml
python MPAP_model_prediciton/prediction_refactored.py

# Or with custom model path
MPAP_MODEL_PATH=path/to/model.tar python MPAP_model_prediciton/prediction_refactored.py
```

## Configuration

All settings are now in `config/config.yaml`:

```yaml
paths:
  train_input_dir: "./MPAP_model_training/train_input"
  valid_input_dir: "./MPAP_model_training/valid_input"
  test_input_dir: "./MPAP_model_training/test_input"
  model_dir: "./MPAP_model_training/models"
  output_dir: "./outputs"

model:
  dim: 75
  layer_gnn: 3
  dropout: 0.012

training:
  batch_size: 128
  learning_rate: 0.0007
  max_epochs: 200
```

## Environment Variables

You can override config settings using environment variables:

```bash
# Custom config file
export MPAP_CONFIG=config/custom_config.yaml

# Custom model path for prediction
export MPAP_MODEL_PATH=./models/best_model.tar

# Then run scripts normally
python MPAP_model_training/training_refactored.py
```

## Output Changes

### Training
- **Logs**: Saved to `logs/mpap.log` (configurable)
- **Models**: Saved to `MPAP_model_training/models/` (configurable)
- **Training metrics**: Saved to `outputs/training_log.txt`

### Prediction
- **Predictions**: Saved to `outputs/predictions.csv`
- **Metrics**: Saved to `outputs/metrics.txt`
- **Console**: Real-time logging of progress

## Migration Steps

### Step 1: Install Package
```bash
pip install -e .
```

### Step 2: Update Config
Edit `config/config.yaml` with your paths and hyperparameters.

### Step 3: Test Refactored Scripts
```bash
# Test data preprocessing
python MPAP_predata/predata_refactored.py --input test_data.txt --output test_output/

# Test training (with small dataset first)
python MPAP_model_training/training_refactored.py

# Test prediction
python MPAP_model_prediciton/prediction_refactored.py
```

### Step 4: Replace Original Scripts (Optional)
Once verified, you can:
1. Backup original scripts
2. Rename refactored versions to original names
3. Or keep both and use refactored versions

## Benefits

1. **No Code Modifications**: Change config file instead of editing code
2. **Reproducibility**: Config files can be version controlled
3. **Logging**: All operations logged for debugging
4. **Error Handling**: Better error messages and exception handling
5. **Portability**: Works on any system (no hardcoded Windows paths)
6. **Maintainability**: Cleaner, more organized code

## Troubleshooting

### Import Errors
If you get import errors, make sure:
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
MPAP_MODEL_PATH=./MPAP_model_prediciton/best-model/0.47535303.tar python MPAP_model_prediciton/prediction_refactored.py
```

## Next Steps

1. Test refactored scripts with your data
2. Adjust `config.yaml` as needed
3. Gradually migrate from original to refactored scripts
4. Consider adding more features (checkpointing, resume training, etc.)

