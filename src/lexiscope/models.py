"""Causal reusable bodies and task heads with explicit padding masks."""

import torch
from torch import nn


def validate_tokens(ids, vocab_size):
    if ids.ndim != 2 or ids.shape[1] == 0:
        raise ValueError("Expected a nonempty batch by sequence tensor")
    if ids.dtype != torch.long or ids.min() < 0 or ids.max() >= vocab_size:
        raise ValueError("Token IDs must be integers inside the tokenizer vocabulary")


class LSTMBody(nn.Module):
    def __init__(self, vocab_size, width=64, layers=2, dropout=0.2, max_length=256):
        super().__init__()
        self.vocab_size, self.width, self.max_length = vocab_size, width, max_length
        self.embedding = nn.Embedding(vocab_size, width)
        self.dropout = nn.Dropout(dropout)
        self.recurrent = nn.LSTM(
            width, width, layers, batch_first=True, dropout=dropout if layers > 1 else 0
        )

    def forward(self, ids, attention_mask=None):
        validate_tokens(ids, self.vocab_size)
        return self.recurrent(self.dropout(self.embedding(ids)))[0]


class TransformerBody(nn.Module):
    def __init__(self, vocab_size, width=64, layers=2, dropout=0.2, max_length=256):
        super().__init__()
        self.vocab_size, self.width, self.max_length = vocab_size, width, max_length
        self.embedding = nn.Embedding(vocab_size, width)
        self.positions = nn.Embedding(max_length, width)
        self.dropout = nn.Dropout(dropout)
        # Construct each layer independently instead of cloning identical parameters.
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    width, 4, width * 4, dropout, batch_first=True
                )
                for _ in range(layers)
            ]
        )

    def forward(self, ids, attention_mask=None):
        validate_tokens(ids, self.vocab_size)
        size = ids.shape[1]
        if size > self.max_length:
            raise ValueError("Sequence exceeds learned positional capacity")
        position = torch.arange(size, device=ids.device).unsqueeze(0)
        hidden = self.dropout(self.embedding(ids) + self.positions(position))
        causal = torch.ones(size, size, dtype=torch.bool, device=ids.device).triu(1)
        padding = None if attention_mask is None else ~attention_mask.bool()
        for layer in self.layers:
            hidden = layer(hidden, src_mask=causal, src_key_padding_mask=padding)
        return hidden


class LanguageModel(nn.Module):
    def __init__(self, body):
        super().__init__()
        self.body = body
        self.head = nn.Linear(body.width, body.vocab_size)

    def forward(self, ids, attention_mask=None):
        return self.head(self.body(ids, attention_mask))


class SentimentModel(nn.Module):
    def __init__(self, body):
        super().__init__()
        self.body = body
        self.head = nn.Linear(body.width, 2)

    def forward(self, ids, attention_mask):
        if attention_mask.shape != ids.shape or not attention_mask.bool().any(1).all():
            raise ValueError("Every sequence needs at least one valid token")
        # Right padding avoids all-masked causal queries before the first valid token.
        if (attention_mask[:, 1:].bool() & ~attention_mask[:, :-1].bool()).any():
            raise ValueError("Use right padding")
        hidden = self.body(ids, attention_mask)
        last = attention_mask.long().sum(1) - 1
        pooled = hidden[torch.arange(len(ids), device=ids.device), last]
        return self.head(pooled)


def parameter_breakdown(model):
    return [
        {"parameter": name, "count": p.numel(), "trainable": p.requires_grad}
        for name, p in model.named_parameters()
    ]
