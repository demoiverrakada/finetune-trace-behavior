from __future__ import annotations

import numpy as np
import scipy.sparse


def sparse_euclidean_nmf(
    matrix: scipy.sparse.spmatrix,
    *,
    rank: int,
    iterations: int = 200,
    seed: int = 20260916,
    epsilon: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit V ~= H @ W with Euclidean multiplicative updates.

    This implements the same nonnegative rank-factorization objective used by
    the pinned Diff Mining toolkit's beta=2, non-orthogonal configuration,
    while retaining the observation matrix in sparse form.
    """
    if rank < 1:
        raise ValueError("rank must be positive")
    if iterations < 1:
        raise ValueError("iterations must be positive")
    value = scipy.sparse.csr_matrix(matrix, dtype=np.float32)
    if value.shape[0] < rank or value.shape[1] < rank:
        raise ValueError("NMF matrix dimensions must be at least rank")
    if value.nnz == 0:
        raise ValueError("NMF matrix must contain positive entries")
    value.data = np.maximum(value.data, 0.0)
    value.eliminate_zeros()
    rng = np.random.default_rng(seed)
    h = rng.random((value.shape[0], rank), dtype=np.float32) + 0.05
    w = rng.random((rank, value.shape[1]), dtype=np.float32) + 0.05
    for _ in range(iterations):
        numerator_h = value @ w.T
        denominator_h = h @ (w @ w.T)
        h *= numerator_h / np.maximum(denominator_h, epsilon)
        numerator_w = np.asarray((value.T @ h).T)
        denominator_w = (h.T @ h) @ w
        w *= numerator_w / np.maximum(denominator_w, epsilon)
        scale = np.maximum(w.sum(axis=1, keepdims=True), epsilon)
        w /= scale
        h *= scale.T
    gram_v = float(np.square(value.data).sum())
    cross = float(np.sum(h * (value @ w.T)))
    gram_h = h.T @ h
    gram_w = w @ w.T
    gram_reconstruction = float(np.sum(gram_h * gram_w))
    residual_squared = max(
        0.0,
        gram_v - 2.0 * cross + gram_reconstruction,
    )
    topic_norms = np.linalg.norm(w, axis=1)
    normalized = w / np.maximum(topic_norms[:, None], epsilon)
    return h, w, {
        "rank": rank,
        "iterations": iterations,
        "seed": seed,
        "frobenius_residual": float(np.sqrt(residual_squared)),
        "topic_cosine_similarity": (normalized @ normalized.T).tolist(),
    }


def top_token_records(
    tokenizer,
    token_ids: np.ndarray,
    ordering_values: np.ndarray,
    average_diffs: np.ndarray,
    count_positive: np.ndarray,
    count_negative: np.ndarray,
) -> list[dict]:
    records = []
    for token_id, ordering, average, positive, negative in zip(
        token_ids,
        ordering_values,
        average_diffs,
        count_positive,
        count_negative,
    ):
        token_id = int(token_id)
        records.append(
            {
                "token_id": token_id,
                "token": tokenizer.decode(
                    [token_id],
                    clean_up_tokenization_spaces=False,
                ),
                "ordering_value": float(ordering),
                "average_logit_difference": float(average),
                "count_positive": int(positive),
                "count_negative": int(negative),
            }
        )
    return records
