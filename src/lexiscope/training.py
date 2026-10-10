"""Sample/token-weighted losses, early stopping and restored checkpoints."""

import random
import time
from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(model, dataset, task, batch_size=32):
    model.eval()
    loss_sum, count, labels, probabilities = 0.0, 0, [], []
    criterion = nn.CrossEntropyLoss(reduction="sum")
    with torch.inference_mode():
        for batch in DataLoader(dataset, batch_size=batch_size):
            if task == "lm":
                x, y = batch
                logits = model(x)
                loss_sum += criterion(logits.flatten(0, 1), y.flatten()).item()
                count += y.numel()
            else:
                x, mask, y = batch
                logits = model(x, mask)
                loss_sum += criterion(logits, y).item()
                count += len(y)
                labels.extend(y.numpy())
                probabilities.extend(logits.softmax(-1)[:, 1].numpy())
    return loss_sum / count, np.asarray(labels), np.asarray(probabilities)


def fit(
    model,
    train,
    validation,
    task,
    identity,
    seed=42,
    epochs=6,
    lr=0.001,
    patience=2,
    batch_size=32,
):
    seed_all(seed)
    loader = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    best, checkpoint, stale, best_epoch = float("inf"), None, 0, 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        total, count = 0.0, 0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            if task == "lm":
                x, y = batch
                logits = model(x)
                loss = criterion(logits.flatten(0, 1), y.flatten())
                units = y.numel()
            else:
                x, mask, y = batch
                loss = criterion(model(x, mask), y)
                units = len(y)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * units
            count += units
        val_loss, _, _ = evaluate(model, validation, task, batch_size)
        history.append(
            {
                "id": identity,
                "seed": seed,
                "epoch": epoch,
                "train_loss": total / count,
                "validation_loss": val_loss,
                "task": task,
            }
        )
        print(
            identity,
            seed,
            epoch,
            round(total / count, 4),
            round(val_loss, 4),
            flush=True,
        )
        if val_loss < best - 1e-5:
            best, checkpoint, stale, best_epoch = (
                val_loss,
                deepcopy(model.state_dict()),
                0,
                epoch,
            )
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(checkpoint)
    model.eval()
    return history, {
        "best_epoch": best_epoch,
        "validation_loss": best,
        "training_seconds": time.perf_counter() - started,
    }
