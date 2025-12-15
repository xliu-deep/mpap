# GitHub 上传文件清单 (中文版)

## 📤 需要上传的文件 (Essential Files)

### 核心代码文件
```
✅ mpap/                          # 主程序包
   ├── __init__.py
   ├── config.py                  # 配置管理
   ├── utils.py                   # 工具函数
   └── data_loader.py             # 数据加载

✅ MPAP_model_training/
   ├── training.py                # 训练脚本 (已重构)
   └── model.py                   # 模型架构

✅ MPAP_model_prediction/
   └── prediction.py             # 预测脚本 (已重构)

✅ MPAP_predata/
   ├── predata.py                 # 数据预处理 (已重构)
   ├── graph_features.py          # 图特征提取
   ├── prints.py                  # 指纹生成
   └── getFeatures.py             # 特征提取工具
```

### 配置文件
```
✅ config/
   └── config.yaml                # 主配置文件

✅ requirements.txt               # Python依赖
✅ environment.yaml               # Conda环境
✅ setup.py                      # 包安装配置
✅ .gitignore                    # Git忽略规则
```

### 文档文件
```
✅ README.md                      # 主文档 (已更新)
✅ LICENSE                        # 许可证
✅ CONTRIBUTING.md                # 贡献指南
✅ QUICK_START.md                 # 快速开始指南
```

### 示例数据 (小文件)
```
✅ MPAP_dataset/
   ├── train.txt                  # 训练数据 (文本格式)
   ├── valid.txt                  # 验证数据 (文本格式)
   └── test.txt                   # 测试数据 (文本格式)
```

## ❌ 不要上传的文件 (Exclude These)

### 备份文件
```
❌ *_backup.py                    # 所有备份文件
❌ *_original_backup.py           # 原始备份
❌ predication.py                 # 旧版本 (已替换为prediction.py)
```

### 大型数据文件 (太大，GitHub限制)
```
❌ *.npy                          # 所有numpy数据文件 (~29MB总计)
❌ MPAP_dataset/dataset-all.xls   # Excel数据文件
❌ MPAP_model_training/*_input/*.npy
❌ MPAP_model_prediction/*_input/*.npy
```

### 模型检查点 (太大)
```
❌ *.tar                          # 模型文件 (~282MB!)
❌ *.pth, *.pt                    # PyTorch模型文件
❌ best-model/                    # 模型目录
```

### 生成的文件
```
❌ logs/                          # 日志目录
❌ outputs/                       # 输出目录
❌ test_preds.txt                # 生成的预测结果
❌ test_input.txt                # 生成的文件
❌ __pycache__/                  # Python缓存
```

### 可选文档 (内部使用，可选)
```
⚠️  OPTIMIZATION_SUMMARY.md       # 优化总结 (可选)
⚠️  MIGRATION_GUIDE.md            # 迁移指南 (可选)
⚠️  REFACTORING_COMPLETE.md       # 重构完成说明 (可选)
⚠️  REPLACEMENT_SUMMARY.md        # 替换总结 (可选)
⚠️  TEST_RESULTS.md               # 测试结果 (可选)
⚠️  test_scripts.py               # 测试脚本 (可选)
```

## 📋 上传前检查清单

### 1. 删除备份文件
```bash
rm MPAP_model_training/training_original_backup.py
rm MPAP_model_prediction/predication_original_backup.py
rm MPAP_model_prediction/predication.py
rm MPAP_predata/predata_original_backup.py
```

### 2. 确认.gitignore已更新
`.gitignore` 应该排除:
- `*.npy` - 数据文件
- `*.tar`, `*.pth`, `*.pt` - 模型文件
- `*_backup.py` - 备份文件
- `__pycache__/` - Python缓存
- `logs/`, `outputs/` - 生成的文件

### 3. 文件大小检查
- ✅ 代码文件: ~50-100 KB
- ✅ 文档: ~50-100 KB  
- ✅ 示例数据: ~100-500 KB
- ✅ **总计应 < 1 MB** (理想大小)

**注意**: 
- `.npy` 文件总计 ~29MB (太大，不要上传)
- 模型文件 ~282MB (太大，不要上传)

## 🚀 上传步骤

### 步骤 1: 清理文件
```bash
# 删除备份文件
find . -name "*_backup.py" -delete
find . -name "*_original_backup.py" -delete

# 删除缓存
find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null
find . -name "*.pyc" -delete
```

### 步骤 2: 初始化Git
```bash
git init
git add .
git commit -m "Initial commit: MPAP model with refactored codebase"
```

### 步骤 3: 创建GitHub仓库并推送
```bash
# 在GitHub上创建新仓库后
git remote add origin https://github.com/yourusername/mpap.git
git branch -M main
git push -u origin main
```

## 📊 文件统计

根据当前项目:
- **代码文件**: 13个Python文件
- **文档文件**: 12个Markdown文件
- **配置文件**: 4个 (yaml, txt, py)

## ⚠️ 重要提醒

1. **大文件处理**: 
   - `.npy` 文件 (~29MB) 和模型文件 (~282MB) 太大
   - 用户需要自己运行 `predata.py` 生成 `.npy` 文件
   - 模型文件可以通过 GitHub Releases 提供

2. **数据文件**:
   - 只上传小的文本文件 (`train.txt`, `valid.txt`, `test.txt`)
   - 不要上传 `.npy` 预处理文件

3. **模型文件**:
   - 模型检查点文件太大，不要上传到代码仓库
   - 可以通过其他方式分享 (GitHub Releases, Google Drive等)

## ✅ 最终检查

上传前确认:
- [ ] 所有 `*_backup.py` 文件已删除
- [ ] `.gitignore` 已更新并排除大文件
- [ ] `README.md` 完整且准确
- [ ] 所有代码文件可以正常导入
- [ ] 仓库大小 < 10MB (理想 < 1MB)

完成这些步骤后，你的项目就可以上传到GitHub了！🎉

