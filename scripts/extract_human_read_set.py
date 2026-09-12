"""Build the applicant reading packet from the corrected n=30 artifacts."""
from __future__ import annotations

import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "READ_ME_TRANSCRIPTS_FIXED_N30.md"


def load(word):
    path = ROOT / f"results/decoder_fix_rerun/e2_{word}_n30.json"
    return json.loads(path.read_text())


lines = [
    "# Corrected n=30 transcripts to read before submission",
    "",
    "These are generated from the parity-validated decoder-fix rerun. Stored labels are",
    "shown for inspection; the applicant must judge whether the labels match the text.",
    "",
]

data = {word: load(word) for word in ("gold", "leaf")}
for word in ("gold", "leaf"):
    for condition in ("sham_batched", "delta"):
        result = data[word]["conditions"][condition]
        labels = result["metrics"]["correct_guess_labels"]
        lines.extend([
            f"## {word} / {condition} — correct-guess probes",
            "",
        ])
        for index, round_ in enumerate(result["rounds"], start=1):
            lines.append(
                f"{index}. [{labels[index - 1]}] Q: "
                f"{round_['correct_guess_prompt']}"
            )
            lines.append(f"   A: {round_['correct_guess_response'].strip()}")
        lines.append("")

population = [
    (word, index + 1)
    for word in ("gold", "leaf")
    for index in range(30)
]
selected = random.Random(20260910).sample(population, 5)
lines.extend([
    "## Seeded qualitative sample",
    "",
    f"Selection: `{selected}`",
    "",
])
for word, path in selected:
    lines.append(f"### {word}, path {path}")
    lines.append("")
    for condition in ("sham_batched", "delta"):
        round_ = data[word]["conditions"][condition]["rounds"][path - 1]
        lines.append(f"- **{condition} final hint:** {round_['warmup_transcript'][-1]['assistant']}")
        lines.append(f"- **{condition} correct response:** {round_['correct_guess_response']}")
    lines.append("")

OUT.write_text("\n".join(lines) + "\n")
print(f"wrote {OUT.relative_to(ROOT)}: {len(lines)} lines")
