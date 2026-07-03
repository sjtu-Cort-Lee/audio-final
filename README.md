# 基于 wav2vec 2.0 的低资源英文 ASR Baseline

这是课程大作业中同学 A 的交接包，项目目标是实现一个基于语音自监督表征的低资源英文 ASR baseline。当前仓库包含训练/评测代码、配置文件、实验结果、论文草稿材料，以及两个可直接评测的最终模型 checkpoint。

GitHub 地址：

```text
https://github.com/wuhaotian0508/audio-final
```

下面所有命令都默认已经进入解压后的项目根目录：

```bash
git clone https://github.com/wuhaotian0508/audio-final.git
cd audio-final
```

## Git LFS 说明

最终模型权重 `model.safetensors` 单文件超过 GitHub 普通 Git 的 100 MB 限制，因此仓库使用 Git LFS 管理模型权重。第一次克隆前建议先安装并启用 Git LFS：

```bash
git lfs install
git clone https://github.com/wuhaotian0508/audio-final.git
cd audio-final
git lfs pull
```

如果没有拉到真实模型文件，`checkpoints/*/model.safetensors` 可能只是很小的 LFS pointer 文件，此时直接执行 `git lfs pull` 即可。

## 仓库内容

本仓库包含代码、配置、实验结果文件、论文草稿材料，以及两个已经训练好的模型目录：

```text
checkpoints/wav2vec2_frozen_1h/
checkpoints/wav2vec2_finetune_1h/
```

其中：

- `wav2vec2_frozen_1h`：冻结 wav2vec 2.0 encoder 的对照模型。
- `wav2vec2_finetune_1h`：微调后的主实验模型。

本仓库没有包含 Hugging Face 的数据/模型缓存、本地 Python 依赖缓存、TensorBoard 日志、中间训练 checkpoint 和 optimizer 状态。这些内容体积较大，可以根据下面命令重新生成。

## 方法说明

基模来源：

- 使用 Hugging Face Transformers 上的 `facebook/wav2vec2-base`。
- 该模型是 wav2vec 2.0 Base，预训练于无标注语音数据。
- 在本项目中，它作为语音自监督 encoder，用于提取上下文语音表征。

ASR 建模方式：

- 使用 `Wav2Vec2ForCTC` 加载预训练 encoder。
- 在 encoder 顶部接一个随机初始化的 CTC 分类头。
- CTC 词表由训练集文本自动构建。
- 音频通过 Hugging Face `datasets` 的 audio pipeline 统一到 16 kHz。
- 文本会被规范化为小写字母、空格和撇号。空格在 CTC 中用 `|` 作为 word delimiter token。

数据设置：

- 数据集：`openslr/librispeech_asr`
- 配置：`clean`
- 训练集：`train.100[:1%]`
- 验证集：`validation[:10%]`
- 测试集：`test[:10%]`

训练设置：

- `wav2vec2_frozen_1h`：冻结整个 wav2vec 2.0 base model，只训练随机初始化的 CTC head。
- `wav2vec2_finetune_1h`：冻结 convolutional feature extractor，微调 Transformer encoder 和 CTC head。
- 使用 fp16 混合精度训练。
- 实验在单张 RTX 4090 上完成。

评测指标：

- WER：Word Error Rate，词错误率。
- CER：Character Error Rate，字符错误率。

## 环境配置

推荐使用 conda：

```bash
conda env create -f environment.yml
conda activate vtex-asr
```

如果当前 shell 没有初始化 conda，可以先执行：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vtex-asr
```

如果机器上已经有可用的 CUDA PyTorch 环境，也可以只安装 Python 依赖：

```bash
pip install -r requirements.txt
```

验证 PyTorch 是否能看到 GPU：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
```

## 缓存数据和原始基模

下面命令会下载/缓存 LibriSpeech 数据和原始 `facebook/wav2vec2-base` 模型：

```bash
bash scripts/prepare_cache.sh configs/wav2vec2_frozen_1h.yaml
```

默认缓存目录为：

```text
.cache/huggingface/
```

如果只想评测包内已经训练好的 checkpoint，也仍然需要在第一次评测时下载对应的测试集数据。

## Smoke Test

在新机器上建议先跑 smoke test，用来检查数据加载、模型加载、训练循环、checkpoint 保存和评测流程是否正常：

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 bash scripts/train.sh configs/wav2vec2_smoke.yaml

PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate.sh \
  configs/wav2vec2_smoke.yaml \
  checkpoints/wav2vec2_smoke
```

smoke test 只训练 2 步，指标没有论文意义，只用于确认代码能跑通。

## 训练冻结 Encoder 的 Baseline

这个实验冻结 wav2vec 2.0 encoder，只训练 CTC head，是一个对照实验：

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 bash scripts/train.sh configs/wav2vec2_frozen_1h.yaml
```

评测冻结模型：

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate.sh \
  configs/wav2vec2_frozen_1h.yaml \
  checkpoints/wav2vec2_frozen_1h
```

主要输出：

```text
checkpoints/wav2vec2_frozen_1h/
results/wav2vec2_frozen_1h/eval_metrics.json
results/wav2vec2_frozen_1h/test_:10%_metrics.json
results/wav2vec2_frozen_1h/eval_predictions.tsv
```

## 训练微调模型

这是主实验。该设置冻结 feature extractor，微调 Transformer encoder 和 CTC head：

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 bash scripts/train.sh configs/wav2vec2_finetune_1h.yaml
```

评测微调模型：

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate.sh \
  configs/wav2vec2_finetune_1h.yaml \
  checkpoints/wav2vec2_finetune_1h
```

主要输出：

```text
checkpoints/wav2vec2_finetune_1h/
results/wav2vec2_finetune_1h/eval_metrics.json
results/wav2vec2_finetune_1h/test_:10%_metrics.json
results/wav2vec2_finetune_1h/eval_predictions.tsv
```

## 直接评测包内模型

如果不想重新训练，只想复用包内已经训练好的模型，可以直接运行：

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate.sh \
  configs/wav2vec2_frozen_1h.yaml \
  checkpoints/wav2vec2_frozen_1h

PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate.sh \
  configs/wav2vec2_finetune_1h.yaml \
  checkpoints/wav2vec2_finetune_1h
```

## 当前实验结果

更完整的结果说明见 [docs/experiment_summary.md](docs/experiment_summary.md)。

| 设置 | 可训练参数量 | Eval WER | Eval CER | Test WER | Test CER |
|---|---:|---:|---:|---:|---:|
| Frozen wav2vec2 + CTC | 24,608 | 0.9965 | 0.7308 | 0.9955 | 0.6998 |
| Fine-tuned wav2vec2 + CTC | 90,195,872 | 0.2917 | 0.0847 | 0.2573 | 0.0747 |

核心结论：

```text
在 1% LibriSpeech 低资源设置下，只训练随机初始化的 CTC head 几乎无法取得有效识别结果；
微调 wav2vec 2.0 encoder 后，WER 和 CER 均显著下降。
```

## 可选：5% 数据量实验

如果同学 B 想补充数据规模消融，可以运行：

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 bash scripts/train.sh configs/wav2vec2_finetune_5pct.yaml

PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate.sh \
  configs/wav2vec2_finetune_5pct.yaml \
  checkpoints/wav2vec2_finetune_5pct
```

这个实验用于比较 `train.100[:1%]` 和 `train.100[:5%]` 的低资源数据规模影响。

## 建议同学 B 做什么

同学 B 不需要重新搭系统，建议重点做下面几件事：

- 复核 `results/` 里的 WER/CER 数字，并整理成论文表格。
- 从 `eval_predictions.tsv` 和 `test_:10%_predictions.tsv` 中挑选代表性错误案例。
- 写错误分析，例如拼写近似错误、专名错误、词边界错误、长句漏词等。
- 如时间允许，运行 `configs/wav2vec2_finetune_5pct.yaml`，补充 1% vs 5% 的数据规模消融。
- 将 frozen vs fine-tuned 对比作为核心消融：低资源 ASR 中，微调 SSL encoder 明显优于只训练 CTC head。

## 目录说明

```text
configs/      实验配置
scripts/      训练、评测、环境检查脚本
src/          训练和评测代码
docs/         交接文档和实验总结
results/      指标和预测文本
checkpoints/  包内包含的 frozen 和 fine-tuned 模型
paper/        ICASSP 风格论文草稿、图表和 PDF
```
