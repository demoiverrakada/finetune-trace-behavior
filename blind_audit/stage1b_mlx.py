from __future__ import annotations

import gc
import math
from pathlib import Path
from typing import Iterable

import numpy as np


BASE_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
DEFAULT_BASE_PATH = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--Qwen--Qwen3-1.7B"
    / "snapshots"
    / BASE_REVISION
)
DEFAULT_ADAPTER_ROOT = (
    Path(__file__).resolve().parents[1]
    / "factorial"
    / "mlx_adapters"
)


def adapter_name_from_endpoint(endpoint: str) -> str | None:
    if endpoint == "base":
        return None
    parts = endpoint.split("::")
    if len(parts) != 4 or parts[0] != "ft":
        raise ValueError(f"invalid Stage 1b endpoint ID: {endpoint!r}")
    _, topic, policy, seed = parts
    if not seed.isdigit():
        raise ValueError(f"invalid Stage 1b endpoint seed: {endpoint!r}")
    return f"v2_{topic}_{policy}_seed{seed}"


def _eos_ids(tokenizer) -> set[int]:
    values = getattr(tokenizer, "eos_token_ids", None)
    if values is None:
        values = getattr(tokenizer, "eos_token_id", None)
    if values is None:
        return set()
    if isinstance(values, int):
        return {int(values)}
    return {int(value) for value in values}


class MlxEndpointRuntime:
    """Single-endpoint MLX runtime with exact token-level accounting."""

    def __init__(
        self,
        *,
        base_path: Path = DEFAULT_BASE_PATH,
        adapter_root: Path = DEFAULT_ADAPTER_ROOT,
        max_prompt_tokens: int = 4096,
    ):
        if max_prompt_tokens < 32:
            raise ValueError("max_prompt_tokens must be at least 32")
        self.base_path = Path(base_path)
        self.adapter_root = Path(adapter_root)
        self.max_prompt_tokens = int(max_prompt_tokens)
        self.endpoint: str | None = None
        self.model = None
        self.tokenizer = None
        self._yes_token_ids: list[int] = []
        self._no_token_ids: list[int] = []

    def load(self, endpoint: str) -> None:
        if self.endpoint == endpoint and self.model is not None:
            return
        self.close()
        import mlx.core as mx
        from mlx_lm import load

        adapter_name = adapter_name_from_endpoint(endpoint)
        adapter_path = (
            self.adapter_root / adapter_name
            if adapter_name is not None
            else None
        )
        if not self.base_path.exists():
            raise FileNotFoundError(self.base_path)
        if adapter_path is not None and not adapter_path.exists():
            raise FileNotFoundError(adapter_path)
        self.model, self.tokenizer = load(
            str(self.base_path),
            adapter_path=(
                str(adapter_path)
                if adapter_path is not None
                else None
            ),
            lazy=False,
        )
        mx.eval(self.model.parameters())
        self.endpoint = endpoint
        self._yes_token_ids = self._single_token_variants(
            ("YES", "Yes", "yes", " yeah", " yep")
        )
        self._no_token_ids = self._single_token_variants(
            ("NO", "No", "no", " nope")
        )
        if not self._yes_token_ids or not self._no_token_ids:
            raise RuntimeError("could not resolve one-token YES/NO variants")

    def close(self) -> None:
        self.endpoint = None
        self.model = None
        self.tokenizer = None
        self._yes_token_ids = []
        self._no_token_ids = []
        gc.collect()
        try:
            import mlx.core as mx

            mx.clear_cache()
        except (ImportError, RuntimeError):
            pass

    def _single_token_variants(self, variants: Iterable[str]) -> list[int]:
        values = []
        for text in variants:
            token_ids = self.tokenizer.encode(
                text,
                add_special_tokens=False,
            )
            if len(token_ids) == 1 and int(token_ids[0]) not in values:
                values.append(int(token_ids[0]))
        return values

    def render_conversation(
        self,
        conversation: list[dict],
        *,
        reserve_tokens: int = 0,
    ) -> list[int]:
        if self.tokenizer is None:
            raise RuntimeError("load an endpoint before rendering prompts")
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        try:
            rendered = self.tokenizer.apply_chat_template(
                conversation,
                enable_thinking=False,
                **kwargs,
            )
        except TypeError:
            rendered = self.tokenizer.apply_chat_template(
                conversation,
                **kwargs,
            )
        token_ids = self.tokenizer.encode(
            rendered,
            add_special_tokens=True,
        )
        return self._truncate_prompt(token_ids, reserve_tokens=reserve_tokens)

    def render_raw(
        self,
        text: str,
        *,
        exact_token_ids: list[int] | None = None,
        reserve_tokens: int = 0,
    ) -> list[int]:
        if self.tokenizer is None:
            raise RuntimeError("load an endpoint before rendering prompts")
        token_ids = (
            [int(value) for value in exact_token_ids]
            if exact_token_ids is not None
            else self.tokenizer.encode(text, add_special_tokens=False)
        )
        return self._truncate_prompt(token_ids, reserve_tokens=reserve_tokens)

    def _truncate_prompt(
        self,
        token_ids: list[int],
        *,
        reserve_tokens: int,
    ) -> list[int]:
        limit = self.max_prompt_tokens - max(0, int(reserve_tokens))
        if limit < 1:
            raise ValueError("generation reserve leaves no prompt capacity")
        if len(token_ids) <= limit:
            return [int(value) for value in token_ids]
        return [int(value) for value in token_ids[-limit:]]

    def generate_from_tokens(
        self,
        prompt_tokens: list[int],
        *,
        max_new_tokens: int,
    ) -> dict:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("load an endpoint before generation")
        if not prompt_tokens:
            raise ValueError("prompt token sequence may not be empty")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        import mlx.core as mx
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        by_position = {}
        eos = _eos_ids(self.tokenizer)
        final = None
        for response in stream_generate(
            self.model,
            self.tokenizer,
            prompt_tokens,
            max_tokens=max_new_tokens,
            sampler=make_sampler(temp=0.0),
        ):
            final = response
            token = int(response.token)
            if token in eos:
                continue
            mx.eval(response.logprobs)
            by_position[int(response.generation_tokens)] = {
                "token_id": token,
                "logprob": float(response.logprobs[token]),
            }
        ordered = [
            by_position[index]
            for index in sorted(by_position)
        ]
        token_ids = [record["token_id"] for record in ordered]
        logprobs = [record["logprob"] for record in ordered]
        return {
            "response": self.tokenizer.decode(
                token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip(),
            "token_ids": token_ids,
            "token_logprobs": logprobs,
            "sum_logprob": float(sum(logprobs)),
            "mean_logprob": (
                float(sum(logprobs) / len(logprobs))
                if logprobs
                else float("-inf")
            ),
            "prompt_tokens": len(prompt_tokens),
            "generation_tokens": len(token_ids),
            "finish_reason": (
                final.finish_reason
                if final is not None
                else "empty"
            ),
            "generation_tps": (
                float(final.generation_tps)
                if final is not None
                else 0.0
            ),
            "peak_memory_gb": (
                float(final.peak_memory)
                if final is not None
                else 0.0
            ),
        }

    def score_continuation(
        self,
        prompt_tokens: list[int],
        continuation_tokens: list[int],
    ) -> dict:
        if self.model is None:
            raise RuntimeError("load an endpoint before scoring")
        if not prompt_tokens:
            raise ValueError("prompt token sequence may not be empty")
        if not continuation_tokens:
            return {
                "token_logprobs": [],
                "sum_logprob": 0.0,
                "mean_logprob": float("-inf"),
                "perplexity": float("inf"),
                "scored_tokens": 0,
            }
        import mlx.core as mx

        combined = [
            *[int(value) for value in prompt_tokens],
            *[int(value) for value in continuation_tokens],
        ]
        inputs = mx.array([combined[:-1]])
        logits = self.model(inputs)[
            0,
            len(prompt_tokens) - 1 :,
            :,
        ].astype(mx.float32)
        logprobs = logits - mx.logsumexp(
            logits,
            axis=-1,
            keepdims=True,
        )
        targets = mx.array(continuation_tokens)[:, None]
        selected = mx.take_along_axis(
            logprobs,
            targets,
            axis=-1,
        ).squeeze(-1)
        mx.eval(selected)
        values = np.asarray(selected, dtype=np.float32)
        mean = float(values.mean())
        return {
            "token_logprobs": values.tolist(),
            "sum_logprob": float(values.sum()),
            "mean_logprob": mean,
            "perplexity": float(math.exp(min(80.0, -mean))),
            "scored_tokens": len(continuation_tokens),
        }

    def next_token_logits(
        self,
        prompt_tokens: list[int],
    ) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("load an endpoint before scoring")
        if not prompt_tokens:
            raise ValueError("prompt token sequence may not be empty")
        import mlx.core as mx

        logits = self.model(mx.array([prompt_tokens]))[
            0,
            -1,
            :,
        ].astype(mx.float32)
        mx.eval(logits)
        return np.asarray(logits, dtype=np.float32)

    def binary_logits(self, conversation: list[dict]) -> dict:
        prompt_tokens = self.render_conversation(conversation)
        logits = self.next_token_logits(prompt_tokens)
        yes = np.logaddexp.reduce(logits[self._yes_token_ids])
        no = np.logaddexp.reduce(logits[self._no_token_ids])
        top_id = int(np.argmax(logits))
        return {
            "yes_logit": float(yes),
            "no_logit": float(no),
            "yes_minus_no": float(yes - no),
            "top_token_id": top_id,
            "top_token": self.tokenizer.decode(
                [top_id],
                clean_up_tokenization_spaces=False,
            ),
            "prompt_tokens": len(prompt_tokens),
        }
