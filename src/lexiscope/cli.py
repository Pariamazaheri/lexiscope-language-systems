"""Run experiments, inspect evidence or predict with a validated checkpoint."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from .data import clean_text
from .models import LSTMBody, SentimentModel, TransformerBody


def predict(root, input_file, output_file, model_name=None):
    root = Path(root)
    meta = json.loads((root / "models/inference.json").read_text())
    identity = model_name or meta["selected_model"]
    frame = pd.read_csv(input_file)
    if "text" not in frame or frame.text.isna().any():
        raise ValueError("Provide a CSV with non-null text")
    text = frame.text.map(clean_text)
    if text.eq("").any():
        raise ValueError("Empty input text")
    if identity == "tfidf_reference":
        p = joblib.load(root / "models/tfidf_reference.joblib").predict_proba(text)[
            :, 1
        ]
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            root / "models/tokenizer", local_files_only=True
        )
        body_class = LSTMBody if identity == "lstm_scratch" else TransformerBody
        model = SentimentModel(body_class(len(tokenizer), **meta["config"]))
        model.load_state_dict(
            torch.load(root / "models" / (identity + "_42.pt"), weights_only=True)
        )
        model.eval()
        values = []
        with torch.inference_mode():
            for start in range(0, len(text), 32):
                encoded = tokenizer(
                    text.iloc[start : start + 32].tolist(),
                    padding="max_length",
                    truncation=True,
                    max_length=128,
                    return_tensors="pt",
                )
                values.extend(
                    model(encoded.input_ids, encoded.attention_mask)
                    .softmax(-1)[:, 1]
                    .numpy()
                )
        p = np.asarray(values)
    frame["positive_probability"] = p
    frame["sentiment"] = np.where(p >= 0.5, "positive", "negative")
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_file, index=False)


def main():
    parser = argparse.ArgumentParser(description="LexiScope language systems benchmark")
    parser.add_argument(
        "command",
        choices=[
            "prepare",
            "train",
            "representations",
            "decode",
            "run",
            "predict",
            "figures",
        ],
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--glove-file", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--model",
        choices=[
            "tfidf_reference",
            "lstm_scratch",
            "transformer_scratch",
            "transformer_transfer",
        ],
    )
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.command == "predict":
        if args.input is None or args.output is None:
            parser.error("predict needs --input and --output")
        predict(args.root, args.input, args.output, args.model)
    elif args.command == "train":
        from .pipeline import run

        print(run(args.root).to_string(index=False))
    elif args.command == "decode":
        from .generation import run

        run(args.root)
    elif args.command == "representations":
        from .embeddings import run

        if args.glove_file is None:
            parser.error("representations needs --glove-file")
        run(args.root, args.glove_file)
    elif args.command == "figures":
        from .visualization import render

        render(args.root)
    else:
        from .sources import prepare

        glove = prepare(args.root, args.glove_file)
        if args.command == "run":
            from .embeddings import run as embeddings
            from .generation import run as generation
            from .pipeline import run as training
            from .visualization import render

            embeddings(args.root, glove)
            generation(args.root)
            training(args.root)
            render(args.root)


if __name__ == "__main__":
    main()
