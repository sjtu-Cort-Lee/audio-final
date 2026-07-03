# 同学 B 操作说明与结果整理

## 1. 推荐操作流程

进入项目根目录：

```bash
cd /data/haotianwu/audio-final
```

README 推荐创建 `vtex-asr` conda 环境：

```bash
conda env create -f environment.yml
conda activate vtex-asr
```

本机实测：`conda env create` 会下载完整 CUDA PyTorch，速度较慢。当前机器已有可用 GPU 环境 `sglang`，我采用了下面方式复用它，并在项目目录补充少量依赖：

```bash
# 已执行：只安装到项目目录 .codex_deps，不污染系统环境
/data/conda/miniconda/envs/sglang/bin/python -m pip install --target .codex_deps jiwer socksio \
  'transformers>=4.41.0,<5.0.0' 'huggingface_hub>=0.34.0,<1.0'
```

验证 GPU：

```bash
PYTHONPATH=$PWD/.codex_deps /data/conda/miniconda/envs/sglang/bin/python - <<'PY'
import torch, transformers, datasets
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')
print('transformers:', transformers.__version__)
print('datasets:', datasets.__version__)
PY
```

本机验证结果：

```text
torch: 2.9.1+cu128
cuda available: True
gpu: NVIDIA A800-SXM4-80GB
transformers: 4.57.6
datasets: 4.8.5
```

## 2. 评测命令

如果需要重新评测包内 checkpoint，可运行：

```bash
export HF_HOME="$PWD/.cache/huggingface"
export HF_DATASETS_CACHE="$PWD/.cache/huggingface/datasets"
export TRANSFORMERS_CACHE="$PWD/.cache/huggingface/transformers"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PWD/.codex_deps"

CUDA_VISIBLE_DEVICES=0 /data/conda/miniconda/envs/sglang/bin/python -B -m src.evaluate \
  --config configs/wav2vec2_frozen_1h.yaml \
  --checkpoint "$PWD/checkpoints/wav2vec2_frozen_1h"

CUDA_VISIBLE_DEVICES=0 /data/conda/miniconda/envs/sglang/bin/python -B -m src.evaluate \
  --config configs/wav2vec2_finetune_1h.yaml \
  --checkpoint "$PWD/checkpoints/wav2vec2_finetune_1h"
```

注意：第一次运行会触发 LibriSpeech 数据下载。本机实测 `datasets` 会准备 48 个数据文件，首次下载预计较久；项目已经包含 `results/` 下的评测结果，因此写报告时可以直接复用这些产物。

## 3. 已复核结果

结果来自项目已有文件：

- `results/wav2vec2_frozen_1h/eval_metrics.json`
- `results/wav2vec2_frozen_1h/test_:10%_metrics.json`
- `results/wav2vec2_finetune_1h/eval_metrics.json`
- `results/wav2vec2_finetune_1h/test_:10%_metrics.json`

| 设置 | Eval WER | Eval CER | Test WER | Test CER |
|---|---:|---:|---:|---:|
| Frozen wav2vec2 + CTC | 0.9965 | 0.7308 | 0.9955 | 0.6998 |
| Fine-tuned wav2vec2 + CTC | 0.2917 | 0.0847 | 0.2573 | 0.0747 |

结论：在 1% LibriSpeech 低资源设置下，仅训练随机初始化的 CTC head 基本无法有效识别；微调 wav2vec 2.0 encoder 后，WER 从约 0.9955 降到 0.2573，CER 从约 0.6998 降到 0.0747，提升非常明显。

## 4. 代表性错误案例

### 4.1 Fine-tuned 模型

Fine-tuned 模型多数错误是局部拼写、同音近似或词边界错误。

| 类型 | Reference | Prediction | 说明 |
|---|---|---|---|
| 拼写近似 | yes i need repose many things have agitated me to day both in mind and body when you return to morrow i shall no longer be the same man | yes i need repose many things have agetated me to day both in mind and body when you return to morrow i shall no longer be the same man | `agitated` 被识别成近似拼写 `agetated` |
| 词尾错误 | bravely and generously has he battled in my behalf and this and more will i dare in his service | bravely and generously has he battled in my behalf and this and more will i dare in his serviscs | `service` 的词尾识别错误 |
| 拼写近似 | it did not beckon or indeed move at all it was as still as the hand of death | it did not beckeon or indeed move at all it was as still as the hand of death | `beckon` 中插入了多余字母 |
| 撇号/规范化 | we won't talk about her any more if you'd rather not we indeed | we won't talk about her any more if youd rather not we indeed | 撇号丢失，语义基本不受影响 |
| 短句整体错误 | hans stirs not | haond sturs night | 短句中专名和关键词都发生混淆 |

### 4.2 Frozen 模型

Frozen 模型输出常为残缺字符序列，说明只训练 CTC head 很难把预训练语音表征直接映射到有效英文文本。

| Reference | Prediction | 说明 |
|---|---|---|
| what was that |  | 完全空输出 |
| mother dear father do you hear me | m | 几乎只输出单个字符 |
| each of us is lashed to some part of the raft | vslm | 输出与原句基本无关 |
| tuesday august eighteenth | tt | 只能保留极少字符 |
| a suffocating smell of nitrogen fills the air it enters the throat it fills the lungs | sfictinm nctn ls ans the rtadtfls lng | 局部保留音素轮廓，但不可读 |

## 5. 可写入报告的分析要点

- Frozen baseline 的 WER 接近 1，说明在低资源条件下，仅训练 CTC 分类头不足以学习稳定的声学到字符映射。
- Fine-tuned 模型显著降低 WER/CER，证明自监督 encoder 的参数适配对低资源 ASR 非常关键。
- Fine-tuned 的主要错误集中在拼写近似、词尾、专名、短句和词边界；这些错误往往保留了较强的发音相似性。
- Frozen 的输出多为短碎片或空串，错误性质不是局部错字，而是整体解码失败。
- 后续如时间允许，可以运行 5% 数据量配置 `configs/wav2vec2_finetune_5pct.yaml`，补充数据规模消融。
