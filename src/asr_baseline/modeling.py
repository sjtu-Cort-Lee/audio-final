from __future__ import annotations

from pathlib import Path

from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)


def _freeze_module(module) -> None:
    for param in module.parameters():
        param.requires_grad = False


def build_processor(vocab_path: str | Path, sampling_rate: int) -> Wav2Vec2Processor:
    tokenizer = Wav2Vec2CTCTokenizer(
        str(vocab_path),
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
    )
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=sampling_rate,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=False,
    )
    return Wav2Vec2Processor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
    )


def build_model(config: dict, processor: Wav2Vec2Processor) -> Wav2Vec2ForCTC:
    model_cfg = config["model"]
    model = Wav2Vec2ForCTC.from_pretrained(
        model_cfg["pretrained_name"],
        vocab_size=len(processor.tokenizer),
        pad_token_id=processor.tokenizer.pad_token_id,
        ctc_loss_reduction="mean",
        ctc_zero_infinity=True,
        ignore_mismatched_sizes=True,
        use_safetensors=True,
    )

    if model_cfg.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()

    last_n = int(model_cfg.get("freeze_encoder_except_last_n", 0) or 0)
    if last_n > 0:
        _freeze_module(model.wav2vec2)
        for layer in model.wav2vec2.encoder.layers[-last_n:]:
            for param in layer.parameters():
                param.requires_grad = True
    elif model_cfg.get("freeze_base_model", False):
        _freeze_module(model.wav2vec2)
    elif model_cfg.get("freeze_feature_encoder", False):
        _freeze_module(model.wav2vec2.feature_extractor)

    return model


def count_trainable_parameters(model: Wav2Vec2ForCTC) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
