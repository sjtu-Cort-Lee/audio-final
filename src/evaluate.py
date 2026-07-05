from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from .asr_baseline.config import load_config, mkdir
from .asr_baseline.data import load_split
from .asr_baseline.metrics import compute_wer_cer
from .asr_baseline.text import normalize_text


def safe_split_name(split: str) -> str:
    name = split.replace("%", "pct")
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")
    return name or "split"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split-key", default="test_split")
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"Loaded config: {args.config}")
    print(f"Dataset: {config['dataset']['name']}")
    print(f"Eval split key: {args.split_key} -> {config['dataset'][args.split_key]}")
    data_cfg = config["dataset"]
    text_column = data_cfg.get("text_column", "text")
    audio_column = data_cfg.get("audio_column", "audio")

    checkpoint = Path(args.checkpoint)
    processor = Wav2Vec2Processor.from_pretrained(checkpoint)
    model = Wav2Vec2ForCTC.from_pretrained(checkpoint)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    dataset = load_split(config, args.split_key)

    def collate_fn(features: list[dict]) -> dict:
        audios = [f[audio_column]["array"] for f in features]
        sampling_rate = features[0][audio_column]["sampling_rate"]
        batch = processor(
            audios,
            sampling_rate=sampling_rate,
            padding=True,
            return_tensors="pt",
        )
        refs = [normalize_text(f[text_column]) for f in features]
        batch["references"] = refs
        return batch

    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["per_device_eval_batch_size"]),
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=int(data_cfg.get("num_proc", 1)),
    )

    predictions: list[str] = []
    references: list[str] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="evaluating"):
            refs = batch.pop("references")
            input_values = batch["input_values"].to(device)
            logits = model(input_values=input_values).logits
            pred_ids = torch.argmax(logits, dim=-1)
            preds = processor.batch_decode(pred_ids)
            predictions.extend(normalize_text(p) for p in preds)
            references.extend(refs)

    metrics = compute_wer_cer(predictions, references)
    results_dir = mkdir(config["results"]["dir"])
    split_name = safe_split_name(data_cfg[args.split_key])
    with (results_dir / f"{split_name}_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with (results_dir / f"{split_name}_predictions.tsv").open("w", encoding="utf-8") as f:
        f.write("reference\tprediction\n")
        for ref, pred in zip(references, predictions):
            f.write(f"{ref}\t{pred}\n")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
