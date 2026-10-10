"""Train predeclared benchmarks and score untouched final partitions."""

import hashlib
import importlib.metadata
import json
import time
from copy import deepcopy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from transformers import AutoTokenizer

from .data import encode_reviews, language_blocks, review_splits
from .decoding import autoregressive
from .evaluation import classification_metrics, paired_bootstrap
from .models import (
    LanguageModel,
    LSTMBody,
    SentimentModel,
    TransformerBody,
    parameter_breakdown,
)
from .training import evaluate, fit, seed_all

CONFIG = {"width": 64, "layers": 2, "dropout": 0.2, "max_length": 256}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run(root):
    root = Path(root)
    report = root / "reports"
    models = root / "models"
    models.mkdir(exist_ok=True)
    report.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(
        root / "data/models/gpt2", local_files_only=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.save_pretrained(models / "tokenizer")
    parts, manifest, audit = review_splits(root / "data/raw/sentiment.csv")
    manifest.to_csv(report / "review_split_manifest.csv", index=False)
    write_json(report / "data_audit.json", audit)
    reviews = {k: encode_reviews(v, tokenizer, 128) for k, v in parts.items()}
    blocks, lm_manifest, lm_audit = language_blocks(
        root / "data/raw/corpus.txt", tokenizer
    )
    lm_manifest.to_csv(report / "lm_split_manifest.csv", index=False)
    write_json(report / "lm_data_audit.json", lm_audit)
    history, parameters, selections, lm_results, generations = [], [], [], [], []
    bodies = {}
    for identity, body_class in [
        ("lstm_lm", LSTMBody),
        ("transformer_lm", TransformerBody),
    ]:
        seed_all(42)
        lm = LanguageModel(body_class(len(tokenizer), **CONFIG))
        parameters.extend({"id": identity, **row} for row in parameter_breakdown(lm))
        initial = autoregressive(
            lm,
            torch.tensor([tokenizer.encode("The old man")]),
            context_limit=64,
            max_new_tokens=15,
            seed=42,
            strategy="sample",
            temperature=0.8,
            top_k=40,
        )
        h, selection = fit(
            lm,
            blocks["train"],
            blocks["validation"],
            "lm",
            identity,
            epochs=4,
            lr=0.001,
            batch_size=16,
        )
        history.extend(h)
        selections.append({"id": identity, "seed": 42, **selection})
        loss, _, _ = evaluate(lm, blocks["test"], "lm", 16)
        lm_results.append(
            {
                "id": identity,
                "test_nll": loss,
                "test_perplexity": float(np.exp(loss)),
                **selection,
            }
        )
        trained = autoregressive(
            lm,
            torch.tensor([tokenizer.encode("The old man")]),
            context_limit=64,
            max_new_tokens=15,
            seed=42,
            strategy="sample",
            temperature=0.8,
            top_k=40,
        )
        generations.append(
            {
                "id": identity,
                "prompt": "The old man",
                "initial": tokenizer.decode(initial[0]),
                "trained": tokenizer.decode(trained[0]),
            }
        )
        bodies[identity] = deepcopy(lm.body)
        torch.save(lm.state_dict(), models / (identity + ".pt"))
    # Preserve the original pretrained body: downstream updates act on deep copies.
    frozen_pretraining = {
        k: v.clone() for k, v in bodies["transformer_lm"].state_dict().items()
    }
    candidates, results, predictions = [], [], []
    for identity, seed in [("lstm_scratch", 42)] + [
        (kind, seed)
        for seed in [42, 43, 44]
        for kind in ["transformer_scratch", "transformer_transfer"]
    ]:
        seed_all(seed)
        head_seed_state = torch.random.get_rng_state()
        if identity == "lstm_scratch":
            body = LSTMBody(len(tokenizer), **CONFIG)
        elif identity == "transformer_scratch":
            body = TransformerBody(len(tokenizer), **CONFIG)
        else:
            body = deepcopy(bodies["transformer_lm"])
        # Matched classification heads for paired transformer comparisons.
        torch.random.set_rng_state(head_seed_state)
        classifier = SentimentModel(body)
        if identity == "transformer_transfer":
            assert all(
                torch.equal(body.state_dict()[k], v)
                for k, v in frozen_pretraining.items()
            )
        parameters.extend(
            {"id": identity, "seed": seed, **row}
            for row in parameter_breakdown(classifier)
        )
        h, selection = fit(
            classifier,
            reviews["train"],
            reviews["validation"],
            "sentiment",
            identity,
            seed=seed,
            epochs=6,
            lr=0.001,
            batch_size=32,
        )
        history.extend(h)
        selections.append({"id": identity, "seed": seed, **selection})
        _, val_y, val_p = evaluate(classifier, reviews["validation"], "sentiment")
        if seed == 42:
            candidates.append(
                {
                    "id": identity,
                    "validation_log_loss": classification_metrics(val_y, val_p)[
                        "log_loss"
                    ],
                }
            )
        _, y, p = evaluate(classifier, reviews["test"], "sentiment")
        results.append(
            {"id": identity, "seed": seed, **classification_metrics(y, p), **selection}
        )
        predictions.append(
            pd.DataFrame(
                {
                    "review_id": parts["test"].review_id.to_numpy(),
                    "id": identity,
                    "seed": seed,
                    "target": y,
                    "probability": p,
                }
            )
        )
        torch.save(
            classifier.state_dict(), models / (identity + "_" + str(seed) + ".pt")
        )
        assert all(
            torch.equal(bodies["transformer_lm"].state_dict()[k], v)
            for k, v in frozen_pretraining.items()
        )
        pd.DataFrame(history).to_csv(report / "training_history.csv", index=False)
        pd.DataFrame(results).to_csv(report / "sentiment_metrics.csv", index=False)
    lexical = make_pipeline(
        TfidfVectorizer(max_features=20000, min_df=2, ngram_range=(1, 2)),
        LogisticRegression(C=2, max_iter=1000, random_state=42),
    )
    start = time.perf_counter()
    lexical.fit(parts["train"].text, parts["train"].label)
    val_p = lexical.predict_proba(parts["validation"].text)[:, 1]
    candidates.append(
        {
            "id": "tfidf_reference",
            "validation_log_loss": classification_metrics(
                parts["validation"].label, val_p
            )["log_loss"],
        }
    )
    p = lexical.predict_proba(parts["test"].text)[:, 1]
    results.append(
        {
            "id": "tfidf_reference",
            "seed": 42,
            **classification_metrics(parts["test"].label.to_numpy(), p),
            "best_epoch": 0,
            "validation_loss": candidates[-1]["validation_log_loss"],
            "training_seconds": time.perf_counter() - start,
        }
    )
    predictions.append(
        pd.DataFrame(
            {
                "review_id": parts["test"].review_id.to_numpy(),
                "id": "tfidf_reference",
                "seed": 42,
                "target": parts["test"].label.to_numpy(),
                "probability": p,
            }
        )
    )
    joblib.dump(lexical, models / "tfidf_reference.joblib")
    all_predictions = pd.concat(predictions, ignore_index=True)
    all_predictions.to_csv(report / "sentiment_predictions.csv", index=False)
    pd.DataFrame(results).to_csv(report / "sentiment_metrics.csv", index=False)
    pd.DataFrame(history).to_csv(report / "training_history.csv", index=False)
    pd.DataFrame(parameters).to_csv(report / "parameter_breakdown.csv", index=False)
    pd.DataFrame(selections).to_csv(report / "checkpoint_selection.csv", index=False)
    pd.DataFrame(lm_results).to_csv(report / "language_model_metrics.csv", index=False)
    pd.DataFrame(candidates).to_csv(report / "validation_candidates.csv", index=False)
    winner = min(candidates, key=lambda x: x["validation_log_loss"])["id"]
    write_json(
        models / "inference.json",
        {
            "selected_model": winner,
            "seed": 42,
            "config": CONFIG,
            "sequence_length": 128,
        },
    )
    write_json(
        report / "selection.json",
        {"selected_model": winner, "criterion": "validation log loss", "seed": 42},
    )
    write_json(report / "custom_generations.json", generations)
    contrasts = []
    for seed in [42, 43, 44]:
        left = all_predictions[
            (all_predictions.id == "transformer_scratch")
            & (all_predictions.seed == seed)
        ]
        right = all_predictions[
            (all_predictions.id == "transformer_transfer")
            & (all_predictions.seed == seed)
        ]
        assert np.array_equal(left.review_id, right.review_id)
        contrasts.append(
            {
                "seed": seed,
                **paired_bootstrap(
                    left.target.to_numpy(),
                    left.probability.to_numpy(),
                    right.probability.to_numpy(),
                ),
            }
        )
    pd.DataFrame(contrasts).to_csv(report / "paired_transfer.csv", index=False)
    write_json(
        report / "run_metadata.json",
        {
            "config": CONFIG,
            "lm_epochs_max": 4,
            "classification_epochs_max": 6,
            "patience": 2,
            "lm_batch_size": 16,
            "sentiment_batch_size": 32,
            "lm_sequence_length": 64,
            "sentiment_sequence_length": 128,
            "downstream_seeds": [42, 43, 44],
            "pretraining_seed": 42,
            "pretraining_scope": (
                "Single shared pretrained checkpoint; "
                "seed contrasts are conditional on it"
            ),
            "vocabulary_size": len(tokenizer),
            "classification_threshold": 0.5,
            "source_hashes": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (root / "src/lexiscope").glob("*.py")
            },
            "input_hashes": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (root / "data/raw").glob("*.txt")
            },
            "sentiment_csv_sha256": hashlib.sha256(
                (root / "data/raw/sentiment.csv").read_bytes()
            ).hexdigest(),
            "versions": {
                p: importlib.metadata.version(p)
                for p in ["torch", "transformers", "numpy", "pandas", "scikit-learn"]
            },
        },
    )
    return pd.DataFrame(results)
