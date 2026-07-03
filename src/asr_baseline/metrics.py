from __future__ import annotations

import numpy as np
from jiwer import cer as jiwer_cer
from jiwer import wer as jiwer_wer

from .text import normalize_text


def compute_wer_cer(predictions: list[str], references: list[str]) -> dict[str, float]:
    preds = [normalize_text(p) for p in predictions]
    refs = [normalize_text(r) for r in references]
    return {
        "wer": float(jiwer_wer(refs, preds)),
        "cer": float(jiwer_cer(refs, preds)),
    }


def decode_predictions(processor, logits: np.ndarray, label_ids: np.ndarray) -> tuple[list[str], list[str]]:
    pred_ids = np.argmax(logits, axis=-1)
    pred_texts = processor.batch_decode(pred_ids)

    label_ids = np.where(label_ids == -100, processor.tokenizer.pad_token_id, label_ids)
    ref_texts = processor.batch_decode(label_ids, group_tokens=False)
    return pred_texts, ref_texts
