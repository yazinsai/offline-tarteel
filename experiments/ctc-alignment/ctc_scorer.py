"""CTC forced-alignment scorer.

Given frame-level log-probabilities and candidate texts,
scores how well each candidate explains the audio using
the CTC forward algorithm (via torch.nn.functional.ctc_loss).
"""
import torch
import torch.nn.functional as F


def score_candidates(
    logits: torch.Tensor,
    candidates: list[dict],
    tokenize_fn: callable,
    blank_id: int,
) -> list[tuple[dict, float]]:
    """Score candidate verses against audio frame logits.

    Args:
        logits: (1, T, V) raw model output
        candidates: list of verse dicts with "text_clean" field
        tokenize_fn: text -> list[int] character index mapping
        blank_id: CTC blank token index

    Returns:
        List of (candidate, score) sorted best-first.
        Score is normalized negative log-likelihood (lower = better match).
    """
    if not candidates:
        return []

    # CTC loss expects (T, N, C) for log_probs
    log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)  # (T, 1, V)
    T = log_probs.size(0)

    # Tokenize all candidates
    encoded = []
    target_lengths = []
    for c in candidates:
        ids = tokenize_fn(c["text_clean"])
        if len(ids) == 0:
            ids = [blank_id]  # fallback for empty text
        encoded.append(ids)
        target_lengths.append(len(ids))

    N = len(candidates)
    # Expand log_probs for batch: (T, 1, V) -> (T, N, V)
    log_probs_batch = log_probs.expand(T, N, -1).contiguous()
    input_lengths = torch.full((N,), T, dtype=torch.long)
    target_lengths_t = torch.tensor(target_lengths, dtype=torch.long)

    # Concatenate targets (CTC loss accepts 1D concatenated targets)
    all_targets = torch.tensor(
        [idx for seq in encoded for idx in seq], dtype=torch.long
    )

    # Batch CTC scoring
    losses = F.ctc_loss(
        log_probs_batch,
        all_targets,
        input_lengths,
        target_lengths_t,
        blank=blank_id,
        reduction="none",
        zero_infinity=True,
    )  # (N,)

    # Normalize by input length to make scores comparable across chunks
    scores = (losses / T).tolist()

    results = list(zip(candidates, scores))
    results.sort(key=lambda x: x[1])  # lower = better
    return results
