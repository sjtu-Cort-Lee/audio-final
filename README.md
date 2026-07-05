# 基于 wav2vec 2.0 的低资源英文 ASR

本项目实现了一个基于语音自监督表征的低资源英文自动语音识别系统。系统使用 `facebook/wav2vec2-base` 作为声学 encoder，在其上训练 CTC 字符分类头，并在 LibriSpeech clean 子集上比较冻结 encoder 与微调 encoder 两种设置。

论文位于 `paper/icassp2026_asr/`，其中 `low_resource_ssl_asr.pdf` 为最终提交 PDF，`low_resource_ssl_asr.tex`、图表、参考文献和 ICASSP 样式文件用于复现论文构建。

## 提交材料结构

```text
README.md                 项目说明、环境配置和复现实验步骤
environment.yml           Conda 环境配置
requirements.txt          Python 依赖列表
configs/                  实验配置文件
scripts/                  训练、评测和缓存准备脚本
src/                      ASR 训练、评测和工具代码
results/                  已保存的指标和预测文本
paper/icassp2026_asr/     ICASSP 2026 风格论文 PDF、源码、图表和参考文献
```

最终提交树不包含 `checkpoints/` 和 Hugging Face 缓存。模型权重和数据集可通过下面的训练脚本重新生成，已报告的指标和预测文本保存在 `results/` 中。

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

## 复现实验

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

生成的评测结果会写入对应的 `results/<experiment_name>/` 目录。`src/evaluate.py` 会将 split 名称清洗为跨平台安全的文件名，例如 `test[:10%]` 会写为 `test_10pct_metrics.json` 和 `test_10pct_predictions.tsv`。

## 编译论文

如需重新生成论文 PDF：

```bash
cd paper/icassp2026_asr
latexmk -xelatex -interaction=nonstopmode low_resource_ssl_asr.tex
```

论文作者区包含小组成员姓名和学号；使用 XeLaTeX 可正确渲染中文姓名。
