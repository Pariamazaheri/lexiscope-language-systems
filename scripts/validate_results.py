"""Independently recompute numerical evidence without data or model downloads."""

import hashlib
import json
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd

from lexiscope.evaluation import classification_metrics, paired_bootstrap
from lexiscope.models import LanguageModel, LSTMBody, SentimentModel, TransformerBody


def main():
    root = Path(__file__).resolve().parents[1]
    report = root / "reports"
    metadata = json.loads((report / "run_metadata.json").read_text())
    manifest = pd.read_csv(report / "review_split_manifest.csv")
    assert manifest.review_id.is_unique
    assert manifest.groupby("text_sha256").split.nunique().max() == 1
    source = pd.read_csv(root / "data/review_source_manifest.csv").set_index(
        "review_id"
    )
    assert (
        source.loc[manifest.review_id].text_sha256.to_list()
        == manifest.text_sha256.to_list()
    )
    assert np.array_equal(source.loc[manifest.review_id].label, manifest.label)
    test = manifest[manifest.split == "test"].set_index("review_id")
    scores = pd.read_csv(report / "sentiment_metrics.csv")
    predictions = pd.read_csv(report / "sentiment_predictions.csv")
    for row in scores.itertuples():
        p = predictions[(predictions.id == row.id) & (predictions.seed == row.seed)]
        assert set(p.review_id) == set(test.index) and p.review_id.is_unique
        assert np.array_equal(test.loc[p.review_id].label, p.target)
        assert p.probability.between(0, 1).all()
        for name, value in classification_metrics(
            p.target.to_numpy(), p.probability.to_numpy()
        ).items():
            np.testing.assert_allclose(value, getattr(row, name), rtol=1e-5, atol=1e-7)
    for row in pd.read_csv(report / "paired_transfer.csv").itertuples():
        left = predictions[
            (predictions.id == "transformer_scratch") & (predictions.seed == row.seed)
        ]
        right = predictions[
            (predictions.id == "transformer_transfer") & (predictions.seed == row.seed)
        ]
        assert np.array_equal(left.review_id, right.review_id)
        contrast = paired_bootstrap(
            left.target.to_numpy(),
            left.probability.to_numpy(),
            right.probability.to_numpy(),
        )
        for name, value in contrast.items():
            np.testing.assert_allclose(value, getattr(row, name), atol=1e-7)
    history = pd.read_csv(report / "training_history.csv")
    for row in pd.read_csv(report / "checkpoint_selection.csv").itertuples():
        h = history[(history.id == row.id) & (history.seed == row.seed)]
        assert row.best_epoch == h.loc[h.validation_loss.idxmin()].epoch
        np.testing.assert_allclose(
            row.validation_loss, h.validation_loss.min(), rtol=1e-6
        )
    lm = pd.read_csv(report / "language_model_metrics.csv")
    np.testing.assert_allclose(np.exp(lm.test_nll), lm.test_perplexity, rtol=1e-7)
    blocks = pd.read_csv(report / "lm_split_manifest.csv").sort_values("start_token")
    assert np.all(
        blocks.start_token.to_numpy()[1:] >= blocks.end_token_exclusive.to_numpy()[:-1]
    )
    assert blocks.block_sha256.is_unique
    candidates = pd.read_csv(report / "validation_candidates.csv")
    selection = json.loads((report / "selection.json").read_text())
    assert (
        selection["selected_model"]
        == candidates.loc[candidates.validation_log_loss.idxmin()].id
    )
    vectors = np.load(report / "contextual_vectors.npz", allow_pickle=False)
    context = pd.read_csv(report / "contextual_similarity.csv")
    for (word, control), rows in context.groupby(["word", "control"]):
        a, b = (
            vectors[word + "_" + control + "_a"],
            vectors[word + "_" + control + "_b"],
        )
        cosine = (a * b).sum(1) / (
            np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        )
        np.testing.assert_allclose(rows.sort_values("layer").cosine, cosine, atol=1e-6)
    assert pd.read_csv(report / "greedy_parity.csv").equal_token_ids.all()
    samples = json.loads((report / "decoding_samples.json").read_text())
    assert all(len(s["token_ids"]) == s["new_tokens"] <= 25 for s in samples)
    parameters = pd.read_csv(report / "parameter_breakdown.csv")
    for (identity, seed), group in parameters.groupby(["id", "seed"], dropna=False):
        cls = LSTMBody if identity.startswith("lstm") else TransformerBody
        body = cls(metadata["vocabulary_size"], **metadata["config"])
        model = (
            LanguageModel(body) if identity.endswith("_lm") else SentimentModel(body)
        )
        assert dict(zip(group.parameter, group["count"])) == {
            n: p.numel() for n, p in model.named_parameters()
        }
    summary = pd.read_csv(report / "parameter_summary.csv")
    for row in summary.itertuples():
        group = parameters[
            (parameters.id == row.id)
            & (parameters.seed.isna() | parameters.seed.eq(42))
        ]
        assert row.total == group["count"].sum()
        lexical = group[group.parameter.str.startswith(("body.embedding", "head."))][
            "count"
        ].sum()
        assert row.embedding_and_head == lexical
        assert row.total == (
            row.embedding_and_head + row.position_parameters + row.core_parameters
        )
        np.testing.assert_allclose(row.embedding_head_fraction, lexical / row.total)
    for name, sha in metadata["source_hashes"].items():
        assert (
            hashlib.sha256((root / "src/lexiscope" / name).read_bytes()).hexdigest()
            == sha
        )
    corpus = json.loads((root / "data/source_metadata.json").read_text())["corpus"]
    assert (
        hashlib.sha256((root / "data/corpus.txt").read_bytes()).hexdigest()
        == corpus["sha256"]
    )
    notebook = nbformat.read(root / "notebooks/language_systems.ipynb", as_version=4)
    cells = [c for c in notebook.cells if c.cell_type == "code"]
    assert cells and all(c.execution_count is not None for c in cells)
    assert not any(o.output_type == "error" for c in cells for o in c.outputs)
    print(f"Verified {len(scores)} classifiers and {len(samples)} decoding runs")


if __name__ == "__main__":
    main()
