"""Reconstruct public source inputs without distributing large models or reviews."""

import hashlib
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download

from .data import clean_text, digest

MODELS = {
    "bert": (
        "google-bert/bert-base-uncased",
        "86b5e0934494bd15c9632b12f734a8a67f723594",
    ),
    "gpt2": ("openai-community/gpt2", "607a30d783dfa663caf39e06633721c8d4cfcd7e"),
    "tinystories": (
        "roneneldan/TinyStories-1M",
        "77f1b168e219585646439073245fe87e56b3023e",
    ),
}


def download(url, path):
    if not Path(path).exists():
        urllib.request.urlretrieve(url, path)


def prepare(root, glove_file=None, include_glove=True):
    root = Path(root)
    raw = root / "data/raw"
    raw.mkdir(parents=True, exist_ok=True)
    if not (raw / "corpus.txt").exists():
        shutil.copyfile(root / "data/corpus.txt", raw / "corpus.txt")
    if not (raw / "sentiment.csv").exists():
        archive = raw / "aclImdb_v1.tar.gz"
        download(
            "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz", archive
        )
        expected = "c40f74a18d3b61f90feba1e17730e0d38e8b97c05fde7008942e91923d1658fe"
        if hashlib.sha256(archive.read_bytes()).hexdigest() != expected:
            raise ValueError("IMDb archive changed; verify provenance before updating")
        manifest = pd.read_csv(root / "data/review_source_manifest.csv")
        wanted = {row.archive_member: row for row in manifest.itertuples()}
        by_member = {}
        # Stream gzip once; random per-review seeks repeatedly decompress the archive.
        with tarfile.open(archive, "r|gz") as handle:
            for member in handle:
                if member.name not in wanted:
                    continue
                row = wanted[member.name]
                text = handle.extractfile(member).read().decode("utf-8")
                if digest(clean_text(text)) != row.text_sha256:
                    raise ValueError("Source review content mismatch")
                by_member[member.name] = text
        rows = [
            {"text": by_member[row.archive_member], "label": row.label}
            for row in manifest.itertuples()
        ]
        pd.DataFrame(rows).to_csv(raw / "sentiment.csv", index=False)
    for folder, (model_id, revision) in MODELS.items():
        destination = root / "data/models" / folder
        tokenizer_ready = (destination / "tokenizer.json").exists()
        weights_ready = any(
            (destination / name).exists()
            for name in ["model.safetensors", "pytorch_model.bin"]
        )
        ready = tokenizer_ready and (folder == "gpt2" or weights_ready)
        if not ready:
            patterns = [
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.txt",
                "vocab.json",
                "merges.txt",
                "special_tokens_map.json",
                "generation_config.json",
            ]
            if folder != "gpt2":
                patterns.extend(["model.safetensors", "pytorch_model.bin"])
            snapshot_download(
                model_id,
                revision=revision,
                local_dir=destination,
                allow_patterns=patterns,
            )
    if include_glove and glove_file is None:
        glove_file = raw / "glove.6B.200d.txt"
        if not glove_file.exists():
            archive = raw / "glove.6B.zip"
            download("https://nlp.stanford.edu/data/glove.6B.zip", archive)
            with zipfile.ZipFile(archive) as handle:
                glove_file.write_bytes(handle.read("glove.6B.200d.txt"))
    return Path(glove_file) if glove_file else None
