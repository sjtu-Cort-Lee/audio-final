# 交接包内容说明

本 zip 包包含课程大作业低资源 ASR baseline 的可复现实验材料。

## 包含内容

- `README.md`：环境配置、训练、评测和复现实验说明。
- `environment.yml`、`requirements.txt`：环境依赖。
- `configs/`：实验配置文件。
- `scripts/`：环境检查、缓存准备、训练、评测入口脚本。
- `src/`：数据加载、模型构建、训练和评测代码。
- `docs/handoff.md`：给组内同学的交接说明。
- `docs/experiment_summary.md`：实验结果表和错误分析。
- `results/`：轻量级结果文件，包括 WER/CER 和预测文本。
- `checkpoints/wav2vec2_frozen_1h/`：冻结 encoder 的对照模型。
- `checkpoints/wav2vec2_finetune_1h/`：微调后的主实验模型。
- `hw.txt`、`group.md`：作业说明和小组初步方案。

## 未包含内容

- `.cache/`：Hugging Face 数据集和模型缓存。
- `checkpoint-600/`、`checkpoint-800/` 等中间 checkpoint。
- `optimizer.pt` 等优化器状态。
- `logs/`：TensorBoard 和训练日志。
- `__pycache__/`：Python 字节码缓存。

未包含的内容都可以根据 `README.md` 中的命令重新生成。最终 frozen 模型和 fine-tuned 模型已经包含在包内。

## 模型目录

```text
checkpoints/wav2vec2_frozen_1h/
checkpoints/wav2vec2_finetune_1h/
```

两个目录都包含 `model.safetensors`、tokenizer 文件、processor 配置、模型配置和训练元数据，可以直接通过 `scripts/evaluate.sh` 加载评测。

## 核心结果

| 设置 | Eval WER | Eval CER | Test WER | Test CER |
|---|---:|---:|---:|---:|
| Frozen wav2vec2 + CTC | 0.9965 | 0.7308 | 0.9955 | 0.6998 |
| Fine-tuned wav2vec2 + CTC | 0.2917 | 0.0847 | 0.2573 | 0.0747 |
