from __future__ import annotations

import inspect

from datasets import Audio, Dataset, load_dataset


def load_split(config: dict, split_name: str) -> Dataset:
    data_cfg = config["dataset"]
    dataset_name = data_cfg["name"]
    if dataset_name == "librispeech_asr":
        dataset_name = "openslr/librispeech_asr"
    dataset_kwargs = {
        "path": dataset_name,
        "split": data_cfg[split_name],
    }
    if data_cfg.get("config"):
        dataset_kwargs["name"] = data_cfg["config"]
    if "trust_remote_code" in inspect.signature(load_dataset).parameters:
        dataset_kwargs["trust_remote_code"] = True

    dataset = load_dataset(**dataset_kwargs)
    dataset = dataset.cast_column(
        data_cfg.get("audio_column", "audio"),
        Audio(sampling_rate=int(data_cfg.get("sampling_rate", 16000))),
    )
    return dataset


def load_train_eval(config: dict) -> tuple[Dataset, Dataset]:
    return load_split(config, "train_split"), load_split(config, "eval_split")
