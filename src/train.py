from __future__ import annotations

import argparse
import inspect
import json
import shutil
from pathlib import Path

import numpy as np
from transformers import Trainer, TrainingArguments, set_seed

from .asr_baseline.collator import DataCollatorCTCWithPadding
from .asr_baseline.config import load_config, mkdir
from .asr_baseline.data import load_train_eval
from .asr_baseline.metrics import compute_wer_cer, decode_predictions
from .asr_baseline.modeling import build_model, build_processor, count_trainable_parameters
from .asr_baseline.text import build_ctc_vocab, normalize_text, text_to_ctc_text


def make_training_args(config: dict) -> TrainingArguments:
    cfg = config["training"]
    signature = inspect.signature(TrainingArguments.__init__)
    kwargs = {
        "output_dir": cfg["output_dir"],
        "logging_dir": cfg["logging_dir"],
        "per_device_train_batch_size": cfg["per_device_train_batch_size"],
        "per_device_eval_batch_size": cfg["per_device_eval_batch_size"],
        "gradient_accumulation_steps": cfg["gradient_accumulation_steps"],
        "learning_rate": cfg["learning_rate"],
        "warmup_steps": cfg["warmup_steps"],
        "max_steps": cfg["max_steps"],
        "save_steps": cfg["save_steps"],
        "logging_steps": cfg["logging_steps"],
        "fp16": cfg.get("fp16", False),
        "save_total_limit": cfg.get("save_total_limit", 2),
        "dataloader_num_workers": cfg.get("dataloader_num_workers", 0),
        "group_by_length": cfg.get("group_by_length", True),
        "report_to": ["tensorboard"],
        "save_strategy": "steps",
        "load_best_model_at_end": False,
        "remove_unused_columns": False,
    }

    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = "steps"
    else:
        kwargs["evaluation_strategy"] = "steps"
    kwargs["eval_steps"] = cfg["eval_steps"]
    kwargs = {k: v for k, v in kwargs.items() if k in signature.parameters}
    return TrainingArguments(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"Loaded config: {args.config}")
    print(f"Dataset: {config['dataset']['name']}")
    print(f"Train split: {config['dataset']['train_split']}")
    print(f"Eval split: {config['dataset']['eval_split']}")
    set_seed(int(config.get("seed", 42)))

    train_dataset, eval_dataset = load_train_eval(config)
    data_cfg = config["dataset"]
    text_column = data_cfg.get("text_column", "text")
    audio_column = data_cfg.get("audio_column", "audio")
    sampling_rate = int(data_cfg.get("sampling_rate", 16000))

    output_dir = mkdir(config["training"]["output_dir"])
    results_dir = mkdir(config["results"]["dir"])
    shutil.copy2(config["_config_path"], results_dir / "config.yaml")

    train_texts = [item[text_column] for item in train_dataset]
    vocab_path = output_dir / "vocab.json"
    build_ctc_vocab(train_texts, vocab_path)
    processor = build_processor(vocab_path, sampling_rate)
    processor.save_pretrained(output_dir)

    def prepare_example(batch: dict) -> dict:
        audio = batch[audio_column]
        batch["input_values"] = processor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
        ).input_values[0]
        batch["labels"] = processor.tokenizer(
            text_to_ctc_text(batch[text_column])
        ).input_ids
        batch["reference_text"] = normalize_text(batch[text_column])
        return batch

    remove_train_columns = train_dataset.column_names
    remove_eval_columns = eval_dataset.column_names
    num_proc = int(data_cfg.get("num_proc", 1))
    train_dataset = train_dataset.map(
        prepare_example,
        remove_columns=remove_train_columns,
        num_proc=num_proc,
    )
    eval_dataset = eval_dataset.map(
        prepare_example,
        remove_columns=remove_eval_columns,
        num_proc=num_proc,
    )

    model = build_model(config, processor)
    total_params, trainable_params = count_trainable_parameters(model)
    param_info = {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
    }
    print(json.dumps(param_info, indent=2))
    with (results_dir / "parameter_count.json").open("w", encoding="utf-8") as f:
        json.dump(param_info, f, indent=2)

    data_collator = DataCollatorCTCWithPadding(processor=processor)

    def compute_metrics(pred) -> dict[str, float]:
        pred_texts, ref_texts = decode_predictions(
            processor,
            np.asarray(pred.predictions),
            np.asarray(pred.label_ids),
        )
        return compute_wer_cer(pred_texts, ref_texts)

    trainer_kwargs = {
        "model": model,
        "args": make_training_args(config),
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
        "compute_metrics": compute_metrics,
    }
    trainer_signature = inspect.signature(Trainer.__init__)
    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = processor
    elif "tokenizer" in trainer_signature.parameters:
        trainer_kwargs["tokenizer"] = processor.feature_extractor
    trainer = Trainer(**trainer_kwargs)

    train_result = trainer.train()
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)

    train_metrics = train_result.metrics
    trainer.save_metrics("train", train_metrics)
    trainer.save_state()

    eval_metrics = trainer.evaluate()
    trainer.save_metrics("eval", eval_metrics)
    with (results_dir / "eval_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(eval_metrics, f, indent=2)

    predictions = trainer.predict(eval_dataset)
    pred_texts, ref_texts = decode_predictions(
        processor,
        np.asarray(predictions.predictions),
        np.asarray(predictions.label_ids),
    )
    with (results_dir / "eval_predictions.tsv").open("w", encoding="utf-8") as f:
        f.write("reference\tprediction\n")
        for ref, pred in zip(ref_texts, pred_texts):
            f.write(f"{normalize_text(ref)}\t{normalize_text(pred)}\n")


if __name__ == "__main__":
    main()
