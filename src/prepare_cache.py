from __future__ import annotations

import argparse

from transformers import AutoConfig

from .asr_baseline.config import load_config
from .asr_baseline.data import load_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    for split_name in ("train_split", "eval_split", "test_split"):
        dataset = load_split(config, split_name)
        print(f"{split_name}: {len(dataset)} examples")

    model_name = config["model"]["pretrained_name"]
    AutoConfig.from_pretrained(model_name)
    print(f"cached model config: {model_name}")


if __name__ == "__main__":
    main()
