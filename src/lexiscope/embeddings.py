"""Static geometry and position-controlled contextual representation audits."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from transformers import AutoModel, AutoTokenizer

GROUPS = {
    "marine": [
        "dolphin",
        "shark",
        "octopus",
        "jellyfish",
        "seahorse",
        "turtle",
        "clownfish",
        "lobster",
        "starfish",
        "whale",
    ],
    "cities": [
        "chicago",
        "paris",
        "tokyo",
        "london",
        "sydney",
        "istanbul",
        "tehran",
        "cairo",
        "dubai",
        "mumbai",
    ],
    "birds": [
        "eagle",
        "sparrow",
        "owl",
        "hummingbird",
        "parrot",
        "penguin",
        "falcon",
        "pigeon",
        "swan",
        "peacock",
    ],
    "countries": [
        "france",
        "germany",
        "italy",
        "spain",
        "brazil",
        "canada",
        "china",
        "japan",
        "india",
        "russia",
    ],
    "music": [
        "jazz",
        "rock",
        "pop",
        "blues",
        "classical",
        "reggae",
        "hiphop",
        "metal",
        "folk",
        "electronic",
    ],
}


class StaticVectors:
    def __init__(self, path):
        words, vectors = [], []
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                word, values = line.rstrip().split(" ", 1)
                vector = np.fromstring(values, sep=" ", dtype=np.float32)
                if len(vector) != 200 or not np.isfinite(vector).all():
                    raise ValueError("Expected finite GloVe 200-dimensional vectors")
                words.append(word)
                vectors.append(vector)
        self.words = words
        self.index = {w: i for i, w in enumerate(words)}
        self.vectors = np.stack(vectors)
        self.norms = np.linalg.norm(self.vectors, axis=1).clip(1e-12)

    def vector(self, word, normalized=False):
        i = self.index[word]
        return self.vectors[i] / self.norms[i] if normalized else self.vectors[i]

    def nearest(self, positive, negative=(), count=5):
        query = sum(self.vector(w, True) for w in positive) - sum(
            self.vector(w, True) for w in negative
        )
        query = query / np.linalg.norm(query)
        scores = (self.vectors @ query) / self.norms
        for w in [*positive, *negative]:
            scores[self.index[w]] = -np.inf
        indices = np.argsort(scores)[-count:][::-1]
        return [(self.words[i], float(scores[i])) for i in indices]


@torch.inference_mode()
def contextual_layers(tokenizer, model, sentence, target):
    start = sentence.lower().index(target.lower())
    encoded = tokenizer(sentence, return_offsets_mapping=True, return_tensors="pt")
    offsets = encoded.pop("offset_mapping")[0]
    indices = [
        i
        for i, (a, b) in enumerate(offsets.tolist())
        if a < start + len(target) and b > start
    ]
    if not indices:
        raise ValueError("Target has no aligned token spans")
    states = model(**encoded, output_hidden_states=True).hidden_states
    vectors = torch.stack([s[0, indices].mean(0) for s in states]).numpy()
    return vectors, indices


def cosine_rows(left, right):
    return np.sum(left * right, axis=-1) / (
        np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1)
    )


def run(root, glove_file):
    root = Path(root)
    report = root / "reports"
    figures = report / "figures"
    torch.set_num_threads(2)
    static = StaticVectors(glove_file)
    queries = [
        ("computer", ["computer"], []),
        ("ocean", ["ocean"], []),
        ("king-man+woman", ["king", "woman"], ["man"]),
        ("paris-france+germany", ["paris", "germany"], ["france"]),
        ("ran-run+swim", ["ran", "swim"], ["run"]),
    ]
    neighbors = [
        {"query": name, "rank": i, "word": w, "cosine": score}
        for name, pos, neg in queries
        for i, (w, score) in enumerate(static.nearest(pos, neg), 1)
    ]
    pd.DataFrame(neighbors).to_csv(report / "static_neighbors.csv", index=False)
    words = [w for group in GROUPS.values() for w in group if w in static.index]
    groups = [g for g, values in GROUPS.items() for w in values if w in static.index]
    vectors = np.stack([static.vector(w, True) for w in words])
    np.savez_compressed(
        report / "static_geometry.npz",
        words=np.array(words),
        groups=np.array(groups),
        vectors=vectors,
    )
    neighborhood = []
    distances = pairwise_distances(vectors, metric="cosine")
    np.fill_diagonal(distances, np.inf)
    same = np.array(groups)[:, None] == np.array(groups)[distances.argsort(1)[:, :5]]
    original_purity = float(same.mean())
    for seed in [42, 43, 44]:
        projection = TSNE(
            n_components=2,
            perplexity=8,
            metric="cosine",
            init="random",
            random_state=seed,
        ).fit_transform(vectors)
        dist2 = pairwise_distances(projection)
        np.fill_diagonal(dist2, np.inf)
        rank = dist2.argsort(1)[:, :5]
        purity = float((np.array(groups)[:, None] == np.array(groups)[rank]).mean())
        overlap = float(
            np.mean(
                [
                    len(set(a) & set(b)) / 5
                    for a, b in zip(distances.argsort(1)[:, :5], rank)
                ]
            )
        )
        neighborhood.append(
            {
                "seed": seed,
                "original_category_neighbor_purity": original_purity,
                "projection_category_neighbor_purity": purity,
                "neighbor_retention": overlap,
            }
        )
        if seed == 42:
            pd.DataFrame(
                {
                    "word": words,
                    "category": groups,
                    "x": projection[:, 0],
                    "y": projection[:, 1],
                }
            ).to_csv(report / "static_projection.csv", index=False)
            fig, ax = plt.subplots(figsize=(12, 8))
            for group in GROUPS:
                indices = [i for i, g in enumerate(groups) if g == group]
                ax.scatter(projection[indices, 0], projection[indices, 1], label=group)
            for w, (x, y) in zip(words, projection):
                ax.annotate(
                    w, (x, y), xytext=(3, 3), textcoords="offset points", fontsize=8
                )
            ax.set(
                title="GloVe semantic geometry | t-SNE seed 42",
                xlabel="t-SNE coordinate 1",
                ylabel="t-SNE coordinate 2",
            )
            ax.legend()
            fig.tight_layout()
            fig.savefig(figures / "static_semantic_map.png", dpi=150)
            plt.close(fig)
    pd.DataFrame(neighborhood).to_csv(report / "projection_stability.csv", index=False)
    tokenizer = AutoTokenizer.from_pretrained(
        root / "data/models/bert", local_files_only=True, use_fast=True
    )
    model = AutoModel.from_pretrained(root / "data/models/bert", local_files_only=True)
    model.eval()
    pairs = [
        (
            "bat",
            "original_positions",
            "The bat flew swiftly through the air.",
            "He grabbed his bat and headed to the baseball field.",
        ),
        (
            "bat",
            "matched_positions",
            "The bat flew through the night.",
            "The bat hit the baseball.",
        ),
        (
            "charge",
            "original_positions",
            "I need to charge my phone before leaving.",
            "He was arrested for the charge of theft.",
        ),
        (
            "charge",
            "matched_positions",
            "They charge the phone before leaving.",
            "They charge the suspect with theft.",
        ),
    ]
    rows, arrays = [], {}
    for word, control, s1, s2 in pairs:
        a, ia = contextual_layers(tokenizer, model, s1, word)
        b, ib = contextual_layers(tokenizer, model, s2, word)
        key = word + "_" + control
        arrays[key + "_a"] = a
        arrays[key + "_b"] = b
        sims = cosine_rows(a, b)
        for layer, score in enumerate(sims):
            rows.append(
                {
                    "word": word,
                    "control": control,
                    "layer": layer,
                    "cosine": float(score),
                    "first_positions": str(ia),
                    "second_positions": str(ib),
                }
            )
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for v, ax, title in [
            (a, axes[0], "First sentence layers"),
            (b, axes[1], "Second sentence layers"),
        ]:
            norm = v / np.linalg.norm(v, axis=1, keepdims=True)
            im = ax.imshow(norm @ norm.T, vmin=-1, vmax=1, cmap="viridis")
            ax.set(title=title, xlabel="Layer", ylabel="Layer")
            fig.colorbar(im, ax=ax, fraction=0.046)
        axes[2].plot(range(len(sims)), sims, marker="o")
        axes[2].set(
            title=word + " | " + control.replace("_", " "),
            xlabel="Layer",
            ylabel="Cross-context cosine",
            ylim=(-0.05, 1.05),
        )
        axes[2].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(figures / (key + ".png"), dpi=150)
        plt.close(fig)
    pd.DataFrame(rows).to_csv(report / "contextual_similarity.csv", index=False)
    np.savez_compressed(report / "contextual_vectors.npz", **arrays)
    meta = {
        "glove_sha256": hashlib.sha256(Path(glove_file).read_bytes()).hexdigest(),
        "glove_words": len(static.words),
        "glove_dimensions": 200,
        "bert_checkpoint_files": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (root / "data/models/bert").iterdir()
            if p.is_file()
        },
        "bert_eval_mode": not model.training,
        "span_pooling": (
            "Mean over WordPiece tokens aligned to the original character span"
        ),
    }
    (report / "embedding_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
