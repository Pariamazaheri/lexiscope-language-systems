"""Numerical and isolation contracts without external downloads."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

from lexiscope import training
from lexiscope.data import language_blocks, review_splits
from lexiscope.decoding import autoregressive, choose, probabilities
from lexiscope.evaluation import paired_bootstrap
from lexiscope.models import LanguageModel, LSTMBody, SentimentModel, TransformerBody


def test_transformer_cannot_read_future_tokens():
    torch.manual_seed(42)
    model = LanguageModel(TransformerBody(20, width=8, layers=2, dropout=0)).eval()
    x = torch.tensor([[1, 2, 3, 4, 5]])
    y = x.clone()
    y[0, 3:] = torch.tensor([11, 12])
    with torch.no_grad():
        torch.testing.assert_close(model(x)[:, :3], model(y)[:, :3])


@pytest.mark.parametrize("body", [LSTMBody, TransformerBody])
def test_right_padding_does_not_change_classification(body):
    model = SentimentModel(body(20, width=8, layers=1, dropout=0)).eval()
    with torch.no_grad():
        compact = model(torch.tensor([[1, 2, 3]]), torch.ones(1, 3))
        padded = model(
            torch.tensor([[1, 2, 3, 19, 19]]), torch.tensor([[1, 1, 1, 0, 0]])
        )
    torch.testing.assert_close(compact, padded, atol=1e-6, rtol=1e-5)


def test_eos_can_be_content_even_when_used_as_padding():
    class Body(nn.Module):
        width = 2

        def forward(self, ids, attention_mask):
            return torch.arange(ids.shape[1]).float()[None, :, None].expand(1, -1, 2)

    model = SentimentModel(Body())
    with torch.no_grad():
        model.head.weight.fill_(1)
        model.head.bias.zero_()
    torch.testing.assert_close(
        model(torch.tensor([[3, 19, 19]]), torch.tensor([[1, 1, 0]])),
        torch.tensor([[2.0, 2.0]]),
    )


@pytest.mark.parametrize("bad", [-1, 20])
def test_invalid_tokens_are_rejected_without_clamping(bad):
    model = LanguageModel(LSTMBody(20, width=8, layers=1))
    with pytest.raises(ValueError, match="vocabulary"):
        model(torch.tensor([[bad]]))


@pytest.mark.parametrize(
    "kwargs",
    [{"temperature": 0}, {"temperature": float("nan")}, {"top_k": 11}, {"top_p": 0}],
)
def test_invalid_sampling_parameters_rejected(kwargs):
    with pytest.raises(ValueError):
        probabilities(torch.zeros(1, 10), **kwargs)


def test_nucleus_retains_the_crossing_token():
    p = probabilities(torch.tensor([[3.0, 2.0, 1.0, 0.0]]), top_p=0.8)
    assert p[0, 0] > 0 and p[0, 1] > 0 and p[0, 2:].eq(0).all()
    torch.testing.assert_close(p.sum(-1), torch.ones(1))


def test_sampling_is_seeded_and_topk_respects_support():
    logits = torch.arange(10).float().unsqueeze(0)
    a = choose(logits, "sample", torch.Generator().manual_seed(42), top_k=3)
    b = choose(logits, "sample", torch.Generator().manual_seed(42), top_k=3)
    assert torch.equal(a, b) and a.item() in [7, 8, 9]


def test_generation_stops_at_eos():
    class Toy(nn.Module):
        def forward(self, x):
            logits = torch.zeros(*x.shape, 5)
            logits[:, :, 4] = 10
            return SimpleNamespace(logits=logits)

    prompt = torch.tensor([[1, 2]])
    assert autoregressive(
        Toy(), prompt, max_new_tokens=10, eos_token_id=4
    ).tolist() == [[1, 2, 4]]
    torch.testing.assert_close(autoregressive(Toy(), prompt, max_new_tokens=0), prompt)


def test_weighted_evaluation_handles_uneven_batches():
    class Toy(nn.Module):
        def forward(self, x, mask):
            return torch.stack([x[:, 0].float(), -x[:, 0].float()], 1)

    x = torch.tensor([[1], [2], [3], [4], [5]])
    mask = torch.ones_like(x)
    y = torch.tensor([0, 0, 1, 1, 1])
    loss, _, _ = training.evaluate(
        Toy(), TensorDataset(x, mask, y), "sentiment", batch_size=2
    )
    assert loss == pytest.approx(
        nn.functional.cross_entropy(Toy()(x, mask), y).item(), rel=1e-6
    )


def test_early_stopping_restores_best_weights(monkeypatch):
    model = LanguageModel(LSTMBody(8, width=8, layers=1, dropout=0))
    data = TensorDataset(torch.tensor([[1, 2], [2, 3]]), torch.tensor([[2, 3], [3, 4]]))
    checkpoints = []
    values = iter([3.0, 2.0, 2.5, 3.0])

    def fake_evaluation(current, *args):
        checkpoints.append(deepcopy(current.state_dict()))
        return next(values), np.array([]), np.array([])

    monkeypatch.setattr(training, "evaluate", fake_evaluation)
    history, selected = training.fit(
        model, data, data, "lm", "toy", epochs=9, patience=2
    )
    assert len(history) == 4 and selected["best_epoch"] == 2
    assert all(torch.equal(v, checkpoints[1][k]) for k, v in model.state_dict().items())


def test_pretraining_survives_downstream_updates():
    original = TransformerBody(20, width=8, layers=1, dropout=0)
    frozen = deepcopy(original.state_dict())
    downstream = SentimentModel(deepcopy(original))
    downstream(torch.tensor([[1, 2, 3]]), torch.ones(1, 3)).sum().backward()
    with torch.no_grad():
        for p in downstream.parameters():
            p.add_(p.grad, alpha=-0.1)
    assert all(torch.equal(v, frozen[k]) for k, v in original.state_dict().items())
    assert any(
        not torch.equal(v, frozen[k]) for k, v in downstream.body.state_dict().items()
    )


def test_duplicates_never_cross_partitions(tmp_path):
    frame = pd.DataFrame(
        {
            "text": [f"review {i}" for i in range(60)],
            "label": [i % 2 for i in range(60)],
        }
    )
    frame = pd.concat([frame, frame.iloc[:2]])
    p = tmp_path / "data.csv"
    frame.to_csv(p, index=False)
    _, manifest, audit = review_splits(p)
    assert audit["duplicates_removed"] == 2
    assert manifest.groupby("text_sha256").split.nunique().max() == 1


def test_lm_blocks_do_not_reuse_positions(tmp_path):
    class Tokenizer:
        def encode(self, text, **kwargs):
            return list(range(len(text)))

    p = tmp_path / "text.txt"
    p.write_text("x" * 1000)
    _, manifest, _ = language_blocks(p, Tokenizer(), length=16, train_limit=500)
    intervals = manifest.sort_values("start_token")
    assert np.all(
        intervals.start_token.to_numpy()[1:]
        >= intervals.end_token_exclusive.to_numpy()[:-1]
    )


def test_paired_bootstrap_uses_same_reviews():
    y = np.array([0, 1] * 20)
    delta = paired_bootstrap(y, 1 - y, y, repeats=100)
    assert delta["accuracy_delta"] == 1 and delta["bootstrap_low"] == 1
