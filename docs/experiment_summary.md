# 实验总结

## 实验设置

任务：基于语音自监督表征的低资源英文 ASR。

数据集：

```text
openslr/librispeech_asr, clean config
train: train.100[:1%]
validation: validation[:10%]
test: test[:10%]
```

模型：

```text
facebook/wav2vec2-base + CTC head
```

对比了两个设置：

- Frozen wav2vec2：冻结完整 wav2vec 2.0 base model，只训练随机初始化的 CTC head。
- Fine-tuned wav2vec2：冻结 convolutional feature extractor，微调 Transformer encoder 和 CTC head。

## 主要结果

| 设置 | 可训练参数量 | Eval WER | Eval CER | Test WER | Test CER |
|---|---:|---:|---:|---:|---:|
| Frozen wav2vec2 + CTC | 24,608 | 0.9965 | 0.7308 | 0.9955 | 0.6998 |
| Fine-tuned wav2vec2 + CTC | 90,195,872 | 0.2917 | 0.0847 | 0.2573 | 0.0747 |

## 结果观察

冻结设置几乎无法学到有效 ASR 行为，因为训练时只优化了随机初始化的 CTC head。在 1% LibriSpeech 低资源设置下，这部分参数量太少，难以完成从 SSL 表征到字符序列的有效映射。

微调 wav2vec 2.0 encoder 后，WER 和 CER 都明显下降，说明预训练语音自监督表征对低资源 ASR 有帮助，但仍需要通过下游监督信号适配具体 ASR 任务。

## 错误类型

微调模型的预测结果整体已经可读，但仍存在几类常见错误：

- 拼写近似错误：例如 `threatened -> thretend`、`blight -> blite`。
- 专名或低频词错误：例如 `hurstwood -> hearstwood`、`cynthia -> sintheia`。
- 发音驱动的替换：例如 `floor -> flor`、`carrie -> carry`。
- 词边界错误：例如 `there would -> therewould`、`anyhow -> any how`。
- 长句中偶发漏词。

这些错误与低资源训练数据较少、没有使用外部语言模型有关。
