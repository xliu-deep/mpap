# GitHub Upload Checklist (上传检查清单)

## Before Upload (上传前检查)

### 1. Remove Unnecessary Files (删除不必要的文件)

```bash
# Delete backup files
rm MPAP_model_training/training_original_backup.py
rm MPAP_model_prediction/predication_original_backup.py
rm MPAP_model_prediction/predication.py
rm MPAP_predata/predata_original_backup.py

# Delete cache directories
find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null
find . -name "*.pyc" -delete
```

### 2. Verify .gitignore (确认.gitignore)

Make sure `.gitignore` includes:
- ✅ `*.npy` - Data files
- ✅ `*.tar`, `*.pth`, `*.pt` - Model files
- ✅ `*_backup.py` - Backup files
- ✅ `__pycache__/` - Python cache
- ✅ `logs/`, `outputs/` - Generated files

### 3. Files to Keep (保留的文件)

**Essential Code:**
- ✅ `mpap/` - All Python files
- ✅ `MPAP_model_training/training.py`
- ✅ `MPAP_model_training/model.py`
- ✅ `MPAP_model_prediction/prediction.py`
- ✅ `MPAP_predata/predata.py`
- ✅ `MPAP_predata/graph_features.py`
- ✅ `MPAP_predata/prints.py`
- ✅ `MPAP_predata/getFeatures.py`

**Configuration:**
- ✅ `config/config.yaml`
- ✅ `requirements.txt`
- ✅ `environment.yaml`
- ✅ `setup.py`

**Documentation:**
- ✅ `README.md`
- ✅ `LICENSE`
- ✅ `CONTRIBUTING.md`
- ✅ `QUICK_START.md`

**Sample Data (Small Files Only):**
- ✅ `MPAP_dataset/train.txt`
- ✅ `MPAP_dataset/valid.txt`
- ✅ `MPAP_dataset/test.txt`

### 4. Files to Exclude (排除的文件)

**Large Data Files:**
- ❌ `MPAP_dataset/dataset-all.xls`
- ❌ All `*.npy` files in `*_input/` directories
- ❌ Model checkpoint files (`*.tar`)

**Backup Files:**
- ❌ All `*_backup.py` files
- ❌ `predication.py` (old version)

**Generated Files:**
- ❌ `test_preds.txt`
- ❌ `test_input.txt`
- ❌ `logs/` directory
- ❌ `outputs/` directory

**Internal Documentation (Optional):**
- ⚠️ `OPTIMIZATION_SUMMARY.md` (can keep or remove)
- ⚠️ `MIGRATION_GUIDE.md` (can keep or remove)
- ⚠️ `REFACTORING_COMPLETE.md` (can keep or remove)
- ⚠️ `REPLACEMENT_SUMMARY.md` (can keep or remove)
- ⚠️ `TEST_RESULTS.md` (can keep or remove)
- ⚠️ `test_scripts.py` (can keep or remove)

## Quick Commands

### Check what will be uploaded:
```bash
git status
```

### See file sizes:
```bash
find . -type f -size +1M -not -path "./.git/*" | sort -h
```

### Clean up:
```bash
# Remove backups
find . -name "*_backup.py" -delete
find . -name "*_original_backup.py" -delete

# Remove cache
find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null
find . -name "*.pyc" -delete
```

## Final Repository Size

After cleanup, your repository should be:
- **Code files**: ~50-100 KB
- **Documentation**: ~50-100 KB
- **Sample data (txt)**: ~100-500 KB
- **Total**: < 1 MB (ideal for GitHub)

Large files (>50MB) should be:
1. Excluded via `.gitignore`
2. Provided via GitHub Releases
3. Or instructions for users to generate them

## Ready to Upload! ✅

Once you've verified the checklist, you're ready to:
1. Initialize git: `git init`
2. Add files: `git add .`
3. Commit: `git commit -m "Initial commit"`
4. Push to GitHub: `git push`

