"""Controlled pretrained decoding comparisons with greedy parity checks."""

import hashlib
import json
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .decoding import autoregressive, probabilities


@torch.inference_mode()
def run(root):
    root = Path(root)
    report = root / "reports"
    torch.set_num_threads(2)
    model = AutoModelForCausalLM.from_pretrained(
        root / "data/models/tinystories", local_files_only=True
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        root / "data/models/tinystories", local_files_only=True
    )
    prompts = [
        ("story", "Once upon a time, there was a little"),
        ("technical", "Artificial Intelligence is"),
        (
            "science_fiction",
            "When the spaceship landed on Mars, the astronauts stepped out and",
        ),
    ]
    output, statistics, parity = [], [], []
    for domain, prompt in prompts:
        encoded = tokenizer(prompt, return_tensors="pt")
        logits = model(**encoded).logits[:, -1]
        p = probabilities(logits)
        statistics.append(
            {
                "domain": domain,
                "entropy_nats": float(-(p * p.clamp_min(1e-12).log()).sum()),
                "top_token_probability": float(p.max()),
                "top_token": tokenizer.decode(p.argmax(-1)),
            }
        )
        manual = autoregressive(
            model,
            encoded.input_ids,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=25,
        )
        for strategy in ["greedy", "beam", "sample"]:
            seeds = [42, 43, 44] if strategy == "sample" else [42]
            for seed in seeds:
                kwargs = {
                    "max_new_tokens": 25,
                    "pad_token_id": tokenizer.eos_token_id,
                    "eos_token_id": tokenizer.eos_token_id,
                    "do_sample": strategy == "sample",
                }
                if strategy == "beam":
                    kwargs.update(num_beams=5)
                if strategy == "sample":
                    kwargs.update(temperature=0.7, top_k=50, top_p=0.9)
                start = time.perf_counter()
                with torch.random.fork_rng():
                    torch.manual_seed(seed)
                    result = model.generate(**encoded, **kwargs)
                seconds = time.perf_counter() - start
                if strategy == "greedy":
                    assert torch.equal(manual, result)
                    parity.append(
                        {
                            "domain": domain,
                            "equal_token_ids": True,
                            "new_tokens": result.shape[1] - encoded.input_ids.shape[1],
                        }
                    )
                new = result[0, encoded.input_ids.shape[1] :]
                grams = [
                    tuple(new[i : i + 3].tolist()) for i in range(max(0, len(new) - 2))
                ]
                repetition = 1 - len(set(grams)) / len(grams) if grams else 0
                output.append(
                    {
                        "domain": domain,
                        "strategy": strategy,
                        "seed": seed,
                        "prompt": prompt,
                        "continuation": tokenizer.decode(new, skip_special_tokens=True),
                        "token_ids": new.tolist(),
                        "new_tokens": len(new),
                        "seconds": seconds,
                        "repeated_trigram_fraction": repetition,
                    }
                )
        for strategy, params in [
            ("temperature", {"temperature": 0.9}),
            ("top_k", {"top_k": 5}),
        ]:
            result = autoregressive(
                model,
                encoded.input_ids,
                eos_token_id=tokenizer.eos_token_id,
                strategy="sample",
                seed=42,
                **params,
            )
            output.append(
                {
                    "domain": domain,
                    "strategy": "custom_" + strategy,
                    "seed": 42,
                    "prompt": prompt,
                    "continuation": tokenizer.decode(
                        result[0, encoded.input_ids.shape[1] :],
                        skip_special_tokens=True,
                    ),
                    "token_ids": result[0, encoded.input_ids.shape[1] :].tolist(),
                    "new_tokens": result.shape[1] - encoded.input_ids.shape[1],
                }
            )
    (report / "decoding_samples.json").write_text(json.dumps(output, indent=2) + "\n")
    pd.DataFrame(statistics).to_csv(report / "prompt_uncertainty.csv", index=False)
    pd.DataFrame(parity).to_csv(report / "greedy_parity.csv", index=False)
    (report / "generation_metadata.json").write_text(
        json.dumps(
            {
                "actual_total_parameters": sum(p.numel() for p in model.parameters()),
                "local_files": {
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (root / "data/models/tinystories").iterdir()
                    if p.is_file()
                },
                "seeded_samples": [42, 43, 44],
                "token_budget": 25,
                "parity_scope": (
                    "One unpadded prompt per call; same EOS and token budget"
                ),
            },
            indent=2,
        )
        + "\n"
    )
