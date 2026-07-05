# 基于 wav2vec 2.0 的低资源英文 ASR

本项目实现了一个基于语音自监督表征的低资源英文自动语音识别系统。系统使用 `facebook/wav2vec2-base` 作为声学 encoder，在其上训练 CTC 字符分类头，并在 LibriSpeech clean 子集上比较冻结 encoder 与微调 encoder 两种设置。

论文位于 `paper/icassp2026_asr/`，其中 `low_resource_ssl_asr.tex` 使用仓库内 `templates/icassp2026/` 提供的 ICASSP 2026 LaTeX 模板组件编写，`low_resource_ssl_asr.pdf` 为提交用 PDF。

## 目录结构

```text
configs/      实验配置文件
scripts/      训练、评测和缓存准备脚本
src/          ASR 训练、评测和工具代码
results/      已保存的指标和预测文本
checkpoints/  训练后模型目录；模型权重由 Git LFS 管理
paper/        ICASSP 2026 风格论文源码、图表和 PDF
templates/    ICASSP 2026 官方模板材料
```

## 克隆仓库

模型权重 `model.safetensors` 由 Git LFS 管理。首次克隆前建议安装并启用 Git LFS：

```bash
git lfs install
git clone https://github.com/wuhaotian0508/audio-final.git
cd audio-final
git lfs pull
```

如果 `checkpoints/*/model.safetensors` 只有一百多字节，说明当前文件仍是 LFS pointer，需要重新执行 `git lfs pull`。

### Windows 克隆问题

旧版本仓库中部分结果文件使用了类似 `test_:10%_metrics.json` 的文件名。冒号 `:` 是 Windows 文件系统非法字符，Git for Windows 会在 checkout 阶段报错：

```text
error: invalid path 'results/wav2vec2_finetune_1h/test_:10%_metrics.json'
fatal: unable to checkout working tree
```

本版本已将这些文件改名为跨平台安全的形式，例如 `test_10pct_metrics.json`、`test_10pct_predictions.tsv` 和 `test_4_metrics.json`，并在 `src/evaluate.py` 中加入 split 名称清洗逻辑，后续评测不会再生成带冒号的文件名。

如果必须处理旧提交，建议在 WSL 或 Linux 环境中 checkout 后执行改名，再提交清理结果：

```bash
git mv 'results/wav2vec2_finetune_1h/test_:10%_metrics.json' results/wav2vec2_finetune_1h/test_10pct_metrics.json
git mv 'results/wav2vec2_finetune_1h/test_:10%_predictions.tsv' results/wav2vec2_finetune_1h/test_10pct_predictions.tsv
git mv 'results/wav2vec2_frozen_1h/test_:10%_metrics.json' results/wav2vec2_frozen_1h/test_10pct_metrics.json
git mv 'results/wav2vec2_frozen_1h/test_:10%_predictions.tsv' results/wav2vec2_frozen_1h/test_10pct_predictions.tsv
git mv 'results/wav2vec2_smoke/test_:4_metrics.json' results/wav2vec2_smoke/test_4_metrics.json
git mv 'results/wav2vec2_smoke/test_:4_predictions.tsv' results/wav2vec2_smoke/test_4_predictions.tsv
```

## 环境配置

推荐使用 conda 创建环境：

```bash
conda env create -f environment.yml
conda activate vtex-asr
```

如果已有可用的 CUDA PyTorch 环境，也可以只安装 Python 依赖：

```bash
pip install -r requirements.txt
```

检查 PyTorch 是否可以使用 GPU：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
```

默认 Hugging Face 缓存目录为项目内 `.cache/huggingface/`。可以先缓存数据集和原始 wav2vec 2.0 模型：

```bash
bash scripts/prepare_cache.sh configs/wav2vec2_frozen_1h.yaml
```

## 运行方式

建议先运行 smoke test 检查数据加载、训练、保存和评测流程：

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 bash scripts/train.sh configs/wav2vec2_smoke.yaml
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate.sh configs/wav2vec2_smoke.yaml checkpoints/wav2vec2_smoke
```

训练冻结 encoder 的 baseline：

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 bash scripts/train.sh configs/wav2vec2_frozen_1h.yaml
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate.sh configs/wav2vec2_frozen_1h.yaml checkpoints/wav2vec2_frozen_1h
```

训练微调 encoder 的主实验模型：

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 bash scripts/train.sh configs/wav2vec2_finetune_1h.yaml
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate.sh configs/wav2vec2_finetune_1h.yaml checkpoints/wav2vec2_finetune_1h
```

生成的评测结果会写入对应的 `results/<experiment_name>/` 目录。

## 编译论文

如需重新生成论文 PDF：

```bash
cd paper/icassp2026_asr
latexmk -xelatex -interaction=nonstopmode low_resource_ssl_asr.tex
```

论文作者区包含小组成员姓名和学号；使用 XeLaTeX 可正确渲染中文姓名。
