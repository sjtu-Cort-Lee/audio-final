# 同学 A 交接文档

## 项目定位

本项目实现一个低资源 ASR baseline，用于课程大作业“基于语音自监督表征的低资源英文 ASR”方向。

核心方法：

```text
LibriSpeech subset -> wav2vec2-base SSL encoder -> CTC head -> text
```

默认 baseline 使用 `facebook/wav2vec2-base`，冻结 base model，只训练 CTC head。这样训练快、显存压力小，适合作为后续实验的共同起点。

数据集使用 Hugging Face 上的 `openslr/librispeech_asr`，`clean` 配置。

## 当前文件结构

```text
configs/
  wav2vec2_smoke.yaml
  wav2vec2_frozen_1h.yaml
  wav2vec2_finetune_1h.yaml
scripts/
  check_gpu_env.sh
  prepare_cache.sh
  train.sh
  evaluate.sh
src/
  train.py
  evaluate.py
  prepare_cache.py
  asr_baseline/
README.md
requirements.txt
environment.yml
```

## 推荐复现流程

```bash
cd <project-root>
bash scripts/check_gpu_env.sh
conda env create -f environment.yml
conda activate vtex-asr
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 bash scripts/train.sh configs/wav2vec2_smoke.yaml
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate.sh configs/wav2vec2_smoke.yaml checkpoints/wav2vec2_smoke
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 bash scripts/train.sh configs/wav2vec2_frozen_1h.yaml
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate.sh configs/wav2vec2_frozen_1h.yaml checkpoints/wav2vec2_frozen_1h
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 bash scripts/train.sh configs/wav2vec2_finetune_1h.yaml
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate.sh configs/wav2vec2_finetune_1h.yaml checkpoints/wav2vec2_finetune_1h
```

如果 GPU 机已有合适环境，可以跳过 `conda env create`，直接 `pip install -r requirements.txt`。

## 结果位置

```text
checkpoints/wav2vec2_frozen_1h/
checkpoints/wav2vec2_finetune_1h/
results/wav2vec2_frozen_1h/eval_metrics.json
results/wav2vec2_frozen_1h/eval_predictions.tsv
results/wav2vec2_frozen_1h/parameter_count.json
results/wav2vec2_finetune_1h/eval_metrics.json
results/wav2vec2_finetune_1h/eval_predictions.tsv
results/wav2vec2_finetune_1h/parameter_count.json
```

论文中可以报告：

- WER
- CER
- 总参数量
- 可训练参数量
- 冻结 encoder 的训练设置

## 已跑通结果

`wav2vec2_frozen_1h` 已完成训练，配置为 `train.100[:1%]`，冻结 wav2vec2 base model，只训练 CTC head。

```text
total_parameters: 94396320
trainable_parameters: 24608
eval_loss: 2.4576
eval_WER: 0.9965
eval_CER: 0.7308
test_WER: 0.9955
test_CER: 0.6998
train_runtime: about 12 minutes on one RTX 4090
```

该结果说明系统流程已经跑通，但冻结整个 SSL encoder 时，仅靠随机初始化的 CTC head 在极低资源数据上识别效果很差。后续实验应优先跑 fine-tuning 设置，作为主要结果或消融对比。

`wav2vec2_finetune_1h` 也已完成训练，配置同样为 `train.100[:1%]`，冻结 feature extractor，微调 Transformer encoder 和 CTC head。

```text
total_parameters: 94396320
trainable_parameters: 90195872
eval_loss: 0.4477
eval_WER: 0.2917
eval_CER: 0.0847
test_WER: 0.2573
test_CER: 0.0747
```

核心对比结果：

| Setting | Trainable Params | Eval WER | Eval CER | Test WER | Test CER |
|---|---:|---:|---:|---:|---:|
| Frozen wav2vec2 + CTC | 24,608 | 0.9965 | 0.7308 | 0.9955 | 0.6998 |
| Fine-tuned wav2vec2 + CTC | 90,195,872 | 0.2917 | 0.0847 | 0.2573 | 0.0747 |

可以写入论文的结论：在 1% LibriSpeech 低资源设置下，仅训练随机初始化的 CTC head 几乎无法完成有效识别；微调 wav2vec2 encoder 后，WER 和 CER 均显著下降，说明预训练语音自监督表征需要通过下游监督信号适配 ASR 任务。

## 后续同学 B 可扩展实验

优先级从高到低：

1. 跑 `configs/wav2vec2_finetune_5pct.yaml`，比较 1% vs 5% 低资源数据规模。
2. 把 `train_split` 从 `train.100[:5%]` 改成 `train.100[:10%]`，继续比较数据规模。
3. 新增 HuBERT 配置，把 `model.pretrained_name` 改成 `facebook/hubert-base-ls960`。
4. 做错误案例分析：从 `eval_predictions.tsv` 中挑选替换、删除、插入错误。

## 论文中可用的一句话方法描述

We build a low-resource ASR baseline by extracting contextual speech representations from a pretrained wav2vec 2.0 encoder and optimizing a randomly initialized CTC prediction head on a small labeled subset of LibriSpeech. We compare frozen and fine-tuned encoder settings using WER and CER.

## 交接说明

包内已经包含 `checkpoints/wav2vec2_frozen_1h/` 和 `checkpoints/wav2vec2_finetune_1h/` 两个最终模型目录，可以直接用于评测。中间 checkpoint、optimizer 状态和 Hugging Face cache 未包含在最终 zip 中。
