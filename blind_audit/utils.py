from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


def compact_json_sha256(payload) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_term(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower())).strip()
