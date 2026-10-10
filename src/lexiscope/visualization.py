"""Decision-relevant views of actual model and representation evidence."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix


def render(root):
    report = Path(root) / "reports"
    folder = report / "figures"
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(folder / (name + ".png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    scores = pd.read_csv(report / "sentiment_metrics.csv")
    order = [
        "lstm_scratch",
        "transformer_scratch",
        "transformer_transfer",
        "tfidf_reference",
    ]
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, identity in enumerate(order):
        group = scores[scores.id == identity]
        values = group.accuracy.to_numpy()
        ax.bar(
            i,
            values.mean(),
            color=["#78909c", "#33658a", "#2a9d8f", "#e9a33b"][i],
            alpha=0.85,
        )
        if len(values) > 1:
            ax.scatter(
                i + np.linspace(-0.12, 0.12, len(values)),
                values,
                color="black",
                s=25,
                zorder=3,
            )
        ax.text(
            i,
            values.max() + 0.01,
            f"{values.mean():.1%}" + f" (n={len(values)})",
            ha="center",
        )
    ax.set(
        xticks=range(4),
        xticklabels=[
            "LSTM scratch",
            "Transformer scratch",
            "Transformer transfer",
            "TF–IDF",
        ],
        ylim=(0, 1),
        ylabel="Final test accuracy",
        title="Practical baseline and repeated downstream training",
    )
    save(fig, "sentiment_comparison")
    contrasts = pd.read_csv(report / "paired_transfer.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    for row in contrasts.itertuples():
        ax.errorbar(
            row.seed,
            row.accuracy_delta * 100,
            yerr=[
                [100 * (row.accuracy_delta - row.bootstrap_low)],
                [100 * (row.bootstrap_high - row.accuracy_delta)],
            ],
            fmt="o",
            color="#2a9d8f",
            capsize=6,
        )
    ax.axhline(0, color="#777777", linestyle="--")
    ax.set(
        xticks=[42, 43, 44],
        xlabel="Paired downstream seed",
        ylabel="Accuracy difference (percentage points)",
        title="Transfer minus scratch | paired review-bootstrap 95% intervals",
    )
    save(fig, "paired_transfer")
    history = pd.read_csv(report / "training_history.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for identity in ["lstm_lm", "transformer_lm"]:
        h = history[history.id == identity]
        axes[0].plot(h.epoch, h.train_loss, label=identity + " train")
        axes[0].plot(
            h.epoch, h.validation_loss, linestyle="--", label=identity + " validation"
        )
    for identity in ["lstm_scratch", "transformer_scratch", "transformer_transfer"]:
        h = history[(history.id == identity) & (history.seed == 42)]
        axes[1].plot(h.epoch, h.validation_loss, marker="o", label=identity)
    axes[0].set(
        title="Next-token learning", xlabel="Epoch", ylabel="NLL per target token"
    )
    axes[1].set(
        title="Sentiment validation | seed 42", xlabel="Epoch", ylabel="NLL per review"
    )
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    save(fig, "learning_curves")
    parameters = pd.read_csv(report / "parameter_breakdown.csv")
    ids = ["lstm_lm", "transformer_lm", "lstm_scratch", "transformer_scratch"]
    table = []
    for identity in ids:
        group = parameters[
            (parameters.id == identity)
            & (parameters.seed.isna() | parameters.seed.eq(42))
        ]
        lexical = group[group.parameter.str.startswith(("body.embedding", "head."))][
            "count"
        ].sum()
        positions = group[group.parameter.str.startswith("body.positions")][
            "count"
        ].sum()
        total = group["count"].sum()
        table.append(
            {
                "id": identity,
                "embedding_and_head": int(lexical),
                "position_parameters": int(positions),
                "core_parameters": int(total - lexical - positions),
                "total": int(total),
                "embedding_head_fraction": lexical / total,
            }
        )
    table = pd.DataFrame(table)
    table.to_csv(report / "parameter_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 4))
    bottom = np.zeros(len(table))
    for column, color in [
        ("embedding_and_head", "#33658a"),
        ("position_parameters", "#e9a33b"),
        ("core_parameters", "#2a9d8f"),
    ]:
        ax.barh(
            table.id,
            table[column] / 1e6,
            left=bottom / 1e6,
            label=column.replace("_", " "),
            color=color,
        )
        bottom += table[column]
    ax.set(
        xlabel="Millions of parameters",
        title="Vocabulary projection dominates compact model size",
    )
    ax.legend(fontsize=8)
    save(fig, "parameter_anatomy")
    predictions = pd.read_csv(report / "sentiment_predictions.csv")
    fig, ax = plt.subplots(figsize=(6, 5))
    for identity in ["tfidf_reference", "transformer_scratch", "transformer_transfer"]:
        p = predictions[(predictions.id == identity) & (predictions.seed == 42)]
        actual, confidence = calibration_curve(
            p.target, p.probability, n_bins=8, strategy="quantile"
        )
        ax.plot(confidence, actual, marker="o", label=identity)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set(
        xlabel="Mean predicted positive probability",
        ylabel="Observed positive frequency",
        title="Held-out reliability | seed 42",
    )
    ax.legend(fontsize=8)
    save(fig, "probability_reliability")
    winner = json.loads((report / "selection.json").read_text())["selected_model"]
    p = predictions[(predictions.id == winner) & (predictions.seed == 42)]
    matrix = confusion_matrix(p.target, p.probability >= 0.5)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(im, ax=ax)
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, str(value), ha="center", va="center")
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
        xlabel="Prediction",
        ylabel="Actual",
        title="Validation-selected model | " + winner,
    )
    save(fig, "selected_model_confusion")
    context = pd.read_csv(report / "contextual_similarity.csv")
    fig, ax = plt.subplots(figsize=(9, 4))
    for (word, control), group in context.groupby(["word", "control"]):
        ax.plot(
            group.layer,
            group.cosine,
            marker="o",
            label=word + " " + control.replace("_", " "),
        )
    ax.set(
        xlabel="BERT layer",
        ylabel="Cross-context cosine",
        ylim=(0, 1.05),
        title="Position controls distinguish input geometry from contextual change",
    )
    ax.legend(fontsize=8)
    save(fig, "contextual_controls")
