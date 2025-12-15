# Quick Start Guide

Get up and running with MPAP in 5 minutes!

## Prerequisites Check

```bash
# Check Python version (need 3.8+)
python --version

# Check if in correct directory
pwd  # Should be in the github/ directory
```

## Step 1: Install Dependencies

```bash
# Option 1: Using conda (recommended)
conda env create -f environment.yaml
conda activate kd

# Option 2: Using pip
pip install -r requirements.txt

# Install package
pip install -e .
```

## Step 2: Verify Installation

```bash
# Test configuration system
python -c "from mpap.config import Config; c = Config(); print('✅ Config works!')"

# Test scripts compile
python -m py_compile MPAP_model_training/training.py MPAP_model_prediciton/prediction.py MPAP_predata/predata.py
echo "✅ Scripts are valid!"
```

## Step 3: Prepare Your Data

### Option A: Use Existing Preprocessed Data

If you already have `.npy` files:
1. Place them in the directories specified in `config/config.yaml`:
   - Training: `MPAP_model_training/train_input/`
   - Validation: `MPAP_model_training/valid_input/`
   - Test: `MPAP_model_training/test_input/`

### Option B: Preprocess Text Data

1. Create a text file with your data (tab-separated):
   ```
   category	psmiles	compound	smiles	average size	water3	logkd	poly_smiles
   PE	[*]C=C[*]	Benzene	C1=CC=CC=C1	100	1	2.5	[*]C=C[*]
   ```

2. Update `config/config.yaml`:
   ```yaml
   paths:
     predata_input: "./path/to/your/data.txt"
     predata_output: "./output_directory/"
   ```

3. Run preprocessing:
   ```bash
   python MPAP_predata/predata.py
   ```

## Step 4: Train the Model

```bash
# Train with default config
python MPAP_model_training/training.py
```

**What happens:**
- Loads config from `config/config.yaml`
- Loads training/validation data
- Trains model for specified epochs
- Saves best model to `MPAP_model_training/models/`
- Logs to `logs/mpap.log`

**Monitor training:**
```bash
# Watch log file
tail -f logs/mpap.log

# Or check training metrics
tail -f outputs/training_log.txt
```

## Step 5: Make Predictions

```bash
# Predict with default model path
python MPAP_model_prediciton/prediction.py
```

**Or specify model:**
```bash
MPAP_MODEL_PATH=./MPAP_model_training/models/best_model_0.123456.tar python MPAP_model_prediciton/prediction.py
```

**Results:**
- Predictions: `outputs/predictions.csv`
- Metrics: `outputs/metrics.txt`

## Common Issues & Solutions

### Issue: "ModuleNotFoundError: No module named 'torch'"
**Solution:**
```bash
pip install torch torchvision torchaudio
# Or
conda install pytorch -c pytorch
```

### Issue: "FileNotFoundError: config/config.yaml"
**Solution:** Make sure you're in the project root directory:
```bash
cd /path/to/github
python MPAP_model_training/training.py
```

### Issue: "CUDA out of memory"
**Solution:** Reduce batch size in `config/config.yaml`:
```yaml
training:
  batch_size: 64  # Reduce from 128
```

### Issue: "No data files found"
**Solution:** Check that data directories exist and contain `.npy` files:
```bash
ls MPAP_model_training/train_input/
# Should show: fingerprints.npy, graph.npy, etc.
```

## Next Steps

- **Customize hyperparameters**: Edit `config/config.yaml`
- **Use different data**: Update paths in `config/config.yaml`
- **Monitor training**: Check `logs/mpap.log` and `outputs/training_log.txt`
- **Analyze results**: Check `outputs/predictions.csv` and `outputs/metrics.txt`

## Getting Help

- Check `README.md` for detailed documentation
- Check `MIGRATION_GUIDE.md` for migration from old scripts
- Check `TEST_RESULTS.md` for validation results
- Open an issue on GitHub for bugs/questions

## Example Workflow

```bash
# 1. Install
pip install -e .
pip install -r requirements.txt

# 2. Preprocess (if needed)
python MPAP_predata/predata.py --input data.txt --output processed/

# 3. Train
python MPAP_model_training/training.py

# 4. Predict
python MPAP_model_prediciton/prediction.py

# 5. Check results
cat outputs/metrics.txt
head outputs/predictions.csv
```

That's it! You're ready to use MPAP! 🎉

