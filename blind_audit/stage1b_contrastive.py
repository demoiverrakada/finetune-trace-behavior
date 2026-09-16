from __future__ import annotations

from collections import Counter
import hashlib
import math
import re

import numpy as np
import scipy.sparse


CONTRASTIVE_CLUSTERS = 8
HASH_DIMENSIONS = 4096


def extraction_prompt(prompt: str, base: str, finetuned: str) -> str:
    return f"""\
Compare two model responses to the same user prompt. State the most specific behavioral
difference caused by Model B. Do not guess training data. If there is no material
difference, say NO_DIFFERENCE.

Use exactly one line beginning with one of:
VARIANT_MORE:
VARIANT_LESS:
NO_DIFFERENCE:

User prompt:
{prompt}

Model A (clean base):
{base}

Model B (anonymous finetuned variant):
{finetuned}
"""


def statement_direction(statement: str) -> str:
    normalized = statement.strip().upper()
    if normalized.startswith("VARIANT_MORE:"):
        return "variant_more"
    if normalized.startswith("VARIANT_LESS:"):
        return "variant_less"
    return "no_difference"


def _tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return words + [
        f"{first}_{second}"
        for first, second in zip(words, words[1:])
    ]


def _hash_feature(token: str, dimensions: int) -> int:
    digest = hashlib.blake2b(
        token.encode("utf-8"),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "big") % dimensions


def hashing_tfidf(
    documents: list[str],
    *,
    dimensions: int = HASH_DIMENSIONS,
) -> scipy.sparse.csr_matrix:
    if not documents:
        raise ValueError("at least one document is required")
    rows = []
    columns = []
    values = []
    document_frequency = Counter()
    tokenized = []
    for document in documents:
        counts = Counter(
            _hash_feature(token, dimensions)
            for token in _tokens(document)
        )
        tokenized.append(counts)
        document_frequency.update(counts)
    for row, counts in enumerate(tokenized):
        for column, count in counts.items():
            rows.append(row)
            columns.append(column)
            tf = 1.0 + math.log(count)
            idf = math.log(
                (1.0 + len(documents))
                / (1.0 + document_frequency[column])
            ) + 1.0
            values.append(tf * idf)
    matrix = scipy.sparse.coo_matrix(
        (values, (rows, columns)),
        shape=(len(documents), dimensions),
        dtype=np.float32,
    ).tocsr()
    norms = np.sqrt(matrix.multiply(matrix).sum(axis=1)).A1
    norms = np.maximum(norms, 1e-8)
    return scipy.sparse.diags(1.0 / norms) @ matrix


def spherical_kmeans(
    matrix: scipy.sparse.csr_matrix,
    *,
    clusters: int = CONTRASTIVE_CLUSTERS,
    seed: int = 20260916,
    iterations: int = 50,
) -> tuple[np.ndarray, np.ndarray, dict]:
    if matrix.shape[0] < clusters:
        raise ValueError("number of documents must be at least clusters")
    rng = np.random.default_rng(seed)
    selected = [int(rng.integers(0, matrix.shape[0]))]
    nearest = np.asarray(
        matrix @ matrix[selected[0]].toarray().ravel()
    ).ravel()
    for _ in range(1, clusters):
        distances = np.maximum(0.0, 1.0 - nearest)
        probabilities = distances * distances
        if probabilities.sum() <= 0:
            remaining = [
                index
                for index in range(matrix.shape[0])
                if index not in selected
            ]
            selected.append(remaining[0])
        else:
            probabilities /= probabilities.sum()
            selected.append(int(rng.choice(matrix.shape[0], p=probabilities)))
        similarity = np.asarray(
            matrix @ matrix[selected[-1]].toarray().ravel()
        ).ravel()
        nearest = np.maximum(nearest, similarity)
    centroids = np.stack(
        [matrix[index].toarray().ravel() for index in selected]
    ).astype(np.float32)
    assignments = np.full(matrix.shape[0], -1, dtype=np.int32)
    completed = 0
    for completed in range(1, iterations + 1):
        similarities = np.asarray(matrix @ centroids.T)
        updated_assignments = np.argmax(similarities, axis=1).astype(np.int32)
        if np.array_equal(assignments, updated_assignments):
            break
        assignments = updated_assignments
        for cluster in range(clusters):
            members = np.flatnonzero(assignments == cluster)
            if len(members) == 0:
                weakest = int(np.argmin(np.max(similarities, axis=1)))
                centroids[cluster] = matrix[weakest].toarray().ravel()
                assignments[weakest] = cluster
                continue
            centroid = np.asarray(matrix[members].sum(axis=0)).ravel()
            norm = np.linalg.norm(centroid)
            centroids[cluster] = centroid / max(norm, 1e-8)
    similarities = np.asarray(matrix @ centroids.T)
    inertia = float(
        np.sum(1.0 - similarities[np.arange(matrix.shape[0]), assignments])
    )
    return assignments, centroids, {
        "clusters": clusters,
        "iterations": completed,
        "cosine_inertia": inertia,
        "cluster_sizes": [
            int(np.count_nonzero(assignments == cluster))
            for cluster in range(clusters)
        ],
    }


def cluster_representatives(
    matrix: scipy.sparse.csr_matrix,
    assignments: np.ndarray,
    centroids: np.ndarray,
    *,
    maximum: int = 20,
) -> list[list[int]]:
    output = []
    for cluster in range(centroids.shape[0]):
        members = np.flatnonzero(assignments == cluster)
        similarities = np.asarray(
            matrix[members] @ centroids[cluster]
        ).ravel()
        order = np.argsort(-similarities, kind="stable")[:maximum]
        output.append([int(members[index]) for index in order])
    return output
