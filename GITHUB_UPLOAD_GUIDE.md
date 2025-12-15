# GitHub Upload Guide (GitHub上传指南)

## Files to Upload (需要上传的文件) ✅

### Core Code Files (核心代码文件)
```
✅ mpap/                    # Main package
   ├── __init__.py
   ├── config.py
   ├── utils.py
   └── data_loader.py

✅ MPAP_model_training/
   ├── training.py          # Main training script
   └── model.py            # Model architecture

✅ MPAP_model_prediction/
   └── prediction.py       # Prediction script

✅ MPAP_predata/
   ├── predata.py          # Data preprocessing
   ├── graph_features.py   # Graph feature extraction
   ├── prints.py           # Fingerprint generation
   └── getFeatures.py      # Feature extraction utilities
```

### Configuration Files (配置文件)
```
✅ config/
   └── config.yaml         # Main configuration

✅ requirements.txt        # Python dependencies
✅ environment.yaml        # Conda environment
✅ setup.py               # Package setup
✅ .gitignore             # Git ignore rules
```

### Documentation (文档)
```
✅ README.md              # Main documentation
✅ LICENSE               # License file
✅ CONTRIBUTING.md       # Contribution guidelines
✅ QUICK_START.md        # Quick start guide
```

### Sample Data (示例数据 - 小文件)
```
✅ MPAP_dataset/
   ├── train.txt         # Training data (text format, small)
   ├── valid.txt         # Validation data (text format, small)
   └── test.txt          # Test data (text format, small)
```

## Files NOT to Upload (不要上传的文件) ❌

### Backup Files (备份文件)
```
❌ MPAP_model_training/training_original_backup.py
❌ MPAP_model_prediction/predication_original_backup.py
❌ MPAP_model_prediction/predication.py  # Old version
❌ MPAP_predata/predata_original_backup.py
```

### Large Data Files (大型数据文件)
```
❌ MPAP_dataset/dataset-all.xls          # Large Excel file
❌ MPAP_model_training/train_input/*.npy # Preprocessed numpy files (large)
❌ MPAP_model_training/valid_input/*.npy
❌ MPAP_model_training/test_input/*.npy
❌ MPAP_model_prediction/train_input/*.npy
❌ MPAP_model_prediction/test_input/*.npy
```

### Model Checkpoints (模型检查点)
```
❌ MPAP_model_prediction/best-model/*.tar # Model files (large, ~MB to GB)
```

### Generated Files (生成的文件)
```
❌ MPAP_model_prediction/test_preds.txt   # Generated predictions
❌ MPAP_model_prediction/test_input.txt  # Generated file
❌ logs/                                  # Log files
❌ outputs/                               # Output files
❌ __pycache__/                          # Python cache
```

### Internal Documentation (内部文档 - 可选)
```
⚠️  OPTIMIZATION_SUMMARY.md      # Internal notes (optional)
⚠️  MIGRATION_GUIDE.md            # Internal notes (optional)
⚠️  REFACTORING_COMPLETE.md       # Internal notes (optional)
⚠️  REPLACEMENT_SUMMARY.md        # Internal notes (optional)
⚠️  TEST_RESULTS.md               # Internal notes (optional)
⚠️  test_scripts.py               # Testing script (optional)
```

### Other Files (其他文件)
```
❌ f1.jpg                          # Image file (if not needed in README)
```

## Updated .gitignore

The `.gitignore` file should exclude:

```gitignore
# Data files
*.npy
*.pkl
*.pickle
*.xls
*.xlsx

# Model files
*.tar
*.pth
*.pt
models/
best-model/

# Outputs
outputs/
results/
logs/
*.log

# Backup files
*_backup.py
*_original_backup.py

# Cache
__pycache__/
*.pyc

# Generated files
test_preds.txt
```

## Recommended Repository Structure

```
github/
├── .gitignore              ✅
├── README.md               ✅
├── LICENSE                 ✅
├── requirements.txt        ✅
├── environment.yaml        ✅
├── setup.py               ✅
├── config/
│   └── config.yaml        ✅
├── mpap/                   ✅
│   ├── __init__.py
│   ├── config.py
│   ├── utils.py
│   └── data_loader.py
├── MPAP_dataset/           ✅ (only .txt files)
│   ├── train.txt
│   ├── valid.txt
│   └── test.txt
├── MPAP_model_training/    ✅
│   ├── training.py
│   └── model.py
├── MPAP_model_prediction/   ✅
│   └── prediction.py
└── MPAP_predata/           ✅
    ├── predata.py
    ├── graph_features.py
    ├── prints.py
    └── getFeatures.py
```

## Quick Upload Checklist

Before uploading to GitHub:

- [ ] Remove all `*_backup.py` files
- [ ] Remove all `*.npy` files (users will generate them)
- [ ] Remove model checkpoints `*.tar` files
- [ ] Remove `__pycache__/` directories
- [ ] Remove `logs/` and `outputs/` directories
- [ ] Update `.gitignore` to exclude data files
- [ ] Keep only essential documentation
- [ ] Verify README.md is complete
- [ ] Test that code works without data files

## File Size Considerations

GitHub has file size limits:
- **100 MB**: Warning for individual files
- **50 MB**: Recommended maximum
- **1 GB**: Hard limit

Large `.npy` files and model checkpoints should:
1. Be excluded via `.gitignore`
2. Be provided via:
   - GitHub Releases (for model files)
   - External storage (Google Drive, etc.)
   - Instructions for users to generate them

## Example Commands

### Clean up before upload:
```bash
# Remove backup files
find . -name "*_backup.py" -delete
find . -name "*_original_backup.py" -delete

# Remove cache
find . -type d -name "__pycache__" -exec rm -r {} +

# Remove large data files (they're in .gitignore but good to verify)
# Note: Don't delete if you need them locally!
```

### Initialize Git and upload:
```bash
# Initialize git (if not already)
git init

# Add all files (respects .gitignore)
git add .

# Commit
git commit -m "Initial commit: MPAP model with refactored codebase"

# Add remote and push
git remote add origin https://github.com/yourusername/mpap.git
git branch -M main
git push -u origin main
```

## Summary

**Upload (上传):**
- ✅ All Python source code
- ✅ Configuration files
- ✅ Documentation (README, LICENSE, etc.)
- ✅ Small text data files (train.txt, valid.txt, test.txt)
- ✅ Requirements and setup files

**Don't Upload (不上传):**
- ❌ Backup files
- ❌ Large .npy data files
- ❌ Model checkpoint files (.tar)
- ❌ Generated outputs
- ❌ Cache files (__pycache__)
- ❌ Log files

This keeps the repository clean, professional, and within GitHub's size limits!

