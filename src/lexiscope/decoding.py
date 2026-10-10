"""Validated sampling and EOS-aware autoregressive generation."""

import torch


def probabilities(logits, temperature=1.0, top_k=0, top_p=1.0):
    if temperature <= 0 or not torch.isfinite(torch.tensor(temperature)):
        raise ValueError("Temperature must be finite and positive")
    if not 0 < top_p <= 1 or not 0 <= top_k <= logits.shape[-1]:
        raise ValueError("Invalid top-k or top-p")
    if logits.ndim != 2 or not torch.isfinite(logits).all():
        raise ValueError("Expected finite batch by vocabulary logits")
    values = logits.clone() / temperature
    if top_k:
        threshold = values.topk(top_k).values[:, -1:]
        values = values.masked_fill(values < threshold, -torch.inf)
    if top_p < 1:
        sorted_values, indices = values.sort(descending=True)
        remove = sorted_values.softmax(-1).cumsum(-1) > top_p
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False
        remove = torch.zeros_like(remove).scatter(1, indices, remove)
        values = values.masked_fill(remove, -torch.inf)
    return values.softmax(-1)


def choose(logits, strategy="greedy", generator=None, **kwargs):
    if strategy == "greedy":
        if logits.ndim != 2 or not torch.isfinite(logits).all():
            raise ValueError("Invalid logits")
        return logits.argmax(-1, keepdim=True)
    if strategy != "sample":
        raise ValueError("Unknown decoding strategy")
    return torch.multinomial(probabilities(logits, **kwargs), 1, generator=generator)


@torch.inference_mode()
def autoregressive(
    model,
    ids,
    max_new_tokens=25,
    eos_token_id=None,
    seed=42,
    strategy="greedy",
    context_limit=None,
    **kwargs,
):
    if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] == 0:
        raise ValueError("Generation accepts one nonempty prompt")
    if max_new_tokens < 0:
        raise ValueError("Token budget must be nonnegative")
    model.eval()
    generator = torch.Generator(device=ids.device).manual_seed(seed)
    result = ids.clone()
    for _ in range(max_new_tokens):
        context = result if context_limit is None else result[:, -context_limit:]
        output = model(context)
        logits = output.logits if hasattr(output, "logits") else output
        token = choose(logits[:, -1], strategy, generator, **kwargs)
        result = torch.cat([result, token], dim=1)
        if eos_token_id is not None and token.item() == eos_token_id:
            break
    return result
