from __future__ import annotations

import contextlib
import gc
import json
import math
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from harness.factorial_causal import _eos_ids, trim_generated_tokens


class PrivateModelOracle:
    """Resolve private Stage-1 endpoint IDs without exposing their metadata."""

    def __init__(
        self,
        root: Path,
        truth_manifest: dict,
        *,
        device: str,
    ):
        self.root = root
        self.truth_manifest = truth_manifest
        self.registry = truth_manifest["endpoint_registry"]
        self.device = device
        self.tokenizer = None
        self.model = None
        self._adapter_names: dict[str, str] = {}
        self._yes_token_ids: list[int] = []
        self._no_token_ids: list[int] = []

    @classmethod
    def from_path(
        cls,
        root: Path,
        truth_path: Path,
        *,
        device: str,
    ) -> "PrivateModelOracle":
        return cls(
            root,
            json.loads(truth_path.read_text()),
            device=device,
        )

    def load(self) -> None:
        if self.model is not None:
            return
        base_record = self.registry["base"]
        tokenizer = AutoTokenizer.from_pretrained(
            base_record["model"],
            revision=base_record["revision"],
            local_files_only=True,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        base = AutoModelForCausalLM.from_pretrained(
            base_record["model"],
            revision=base_record["revision"],
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            local_files_only=True,
        ).to(self.device)

        adapter_endpoints = [
            endpoint
            for endpoint, record in sorted(self.registry.items())
            if record["kind"] == "adapter"
        ]
        if not adapter_endpoints:
            raise RuntimeError("private registry contains no adapters")
        first = adapter_endpoints[0]
        first_name = "anonymous_000"
        model = PeftModel.from_pretrained(
            base,
            str(self.root / self.registry[first]["adapter_path"]),
            adapter_name=first_name,
            local_files_only=True,
        ).to(self.device)
        self._adapter_names[first] = first_name
        for index, endpoint in enumerate(adapter_endpoints[1:], start=1):
            adapter_name = f"anonymous_{index:03d}"
            model.load_adapter(
                str(self.root / self.registry[endpoint]["adapter_path"]),
                adapter_name=adapter_name,
                local_files_only=True,
            )
            self._adapter_names[endpoint] = adapter_name
        model.eval()
        model.requires_grad_(False)

        self.tokenizer = tokenizer
        self.model = model
        self._yes_token_ids = self._single_token_variants(
            ("YES", "Yes", "yes", " yeah", " yep")
        )
        self._no_token_ids = self._single_token_variants(
            ("NO", "No", "no", " nope")
        )
        if not self._yes_token_ids or not self._no_token_ids:
            raise RuntimeError("could not resolve one-token YES/NO variants")

    def _single_token_variants(self, variants: tuple[str, ...]) -> list[int]:
        token_ids = []
        for text in variants:
            encoded = self.tokenizer.encode(
                text,
                add_special_tokens=False,
            )
            if len(encoded) == 1 and encoded[0] not in token_ids:
                token_ids.append(int(encoded[0]))
        return token_ids

    @contextlib.contextmanager
    def activate(self, endpoint: str):
        self.load()
        if endpoint not in self.registry:
            raise KeyError(f"unknown private endpoint {endpoint!r}")
        if self.registry[endpoint]["kind"] == "base":
            with self.model.disable_adapter():
                yield
            return
        self.model.set_adapter(self._adapter_names[endpoint])
        yield

    def _render_batch(self, conversations: list[list[dict]]) -> dict:
        kwargs = {
            "add_generation_prompt": True,
            "return_tensors": "pt",
            "return_dict": True,
            "padding": True,
        }
        try:
            encoded = self.tokenizer.apply_chat_template(
                conversations,
                enable_thinking=False,
                **kwargs,
            )
        except TypeError:
            encoded = self.tokenizer.apply_chat_template(
                conversations,
                **kwargs,
            )
        return {
            key: value.to(self.device)
            for key, value in encoded.items()
        }

    def render_conversation_tokens(
        self,
        conversation: list[dict],
        *,
        reserve_tokens: int = 0,
        max_prompt_tokens: int = 4096,
    ) -> list[int]:
        self.load()
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
        limit = int(max_prompt_tokens) - max(0, int(reserve_tokens))
        if limit < 1:
            raise ValueError("generation reserve leaves no prompt capacity")
        return [int(value) for value in token_ids[-limit:]]

    def render_raw_tokens(
        self,
        text: str,
        *,
        exact_token_ids: list[int] | None = None,
        reserve_tokens: int = 0,
        max_prompt_tokens: int = 4096,
    ) -> list[int]:
        self.load()
        token_ids = (
            [int(value) for value in exact_token_ids]
            if exact_token_ids is not None
            else self.tokenizer.encode(text, add_special_tokens=False)
        )
        limit = int(max_prompt_tokens) - max(0, int(reserve_tokens))
        if limit < 1:
            raise ValueError("generation reserve leaves no prompt capacity")
        return token_ids[-limit:]

    def _token_batch(
        self,
        token_rows: list[list[int]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not token_rows or any(not row for row in token_rows):
            raise ValueError("token rows must be non-empty")
        width = max(len(row) for row in token_rows)
        input_rows = []
        mask_rows = []
        for row in token_rows:
            padding = width - len(row)
            input_rows.append(
                [self.tokenizer.pad_token_id] * padding
                + [int(value) for value in row]
            )
            mask_rows.append([0] * padding + [1] * len(row))
        return (
            torch.tensor(
                input_rows,
                dtype=torch.long,
                device=self.device,
            ),
            torch.tensor(
                mask_rows,
                dtype=torch.long,
                device=self.device,
            ),
        )

    @torch.no_grad()
    def score_token_batch(
        self,
        endpoint: str,
        prompt_rows: list[list[int]],
        continuation_rows: list[list[int]],
    ) -> list[dict]:
        if len(prompt_rows) != len(continuation_rows):
            raise ValueError("prompt and continuation batch sizes differ")
        if not prompt_rows:
            return []
        self.load()
        results = [
            {
                "token_logprobs": [],
                "sum_logprob": 0.0,
                "mean_logprob": float("-inf"),
                "perplexity": float("inf"),
                "scored_tokens": 0,
            }
            for _ in prompt_rows
        ]
        active = [
            index
            for index, continuation in enumerate(continuation_rows)
            if continuation
        ]
        if not active:
            return results
        model_rows = [
            [
                *[int(value) for value in prompt_rows[index]],
                *[int(value) for value in continuation_rows[index][:-1]],
            ]
            for index in active
        ]
        input_ids, attention_mask = self._token_batch(model_rows)
        maximum_continuation = max(
            len(continuation_rows[index])
            for index in active
        )
        with self.activate(endpoint):
            logits = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                logits_to_keep=maximum_continuation,
            ).logits
        for row_index, result_index in enumerate(active):
            continuation = continuation_rows[result_index]
            count = len(continuation)
            row_logits = logits[row_index, -count:, :].float()
            targets = torch.tensor(
                continuation,
                dtype=torch.long,
                device=self.device,
            )
            values = (
                row_logits.gather(1, targets[:, None]).squeeze(1)
                - torch.logsumexp(row_logits, dim=-1)
            )
            array = values.cpu().numpy().astype(np.float32)
            mean = float(array.mean())
            results[result_index] = {
                "token_logprobs": array.tolist(),
                "sum_logprob": float(array.sum()),
                "mean_logprob": mean,
                "perplexity": float(math.exp(min(80.0, -mean))),
                "scored_tokens": count,
            }
        return results

    @torch.no_grad()
    def generate_token_batch(
        self,
        endpoint: str,
        prompt_rows: list[list[int]],
        *,
        max_new_tokens: int,
        include_scores: bool = False,
    ) -> list[dict]:
        if not prompt_rows:
            return []
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        self.load()
        input_ids, attention_mask = self._token_batch(prompt_rows)
        prompt_width = input_ids.shape[1]
        with self.activate(endpoint):
            generated = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        eos_ids = _eos_ids(self.model)
        raw_rows = [
            row.tolist()
            for row in generated[:, prompt_width:]
        ]
        token_rows = [
            trim_generated_tokens(
                row,
                eos_ids=eos_ids,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            for row in raw_rows
        ]
        scores = (
            self.score_token_batch(endpoint, prompt_rows, token_rows)
            if include_scores
            else [None] * len(prompt_rows)
        )
        outputs = []
        for prompt, raw, token_ids, score in zip(
            prompt_rows,
            raw_rows,
            token_rows,
            scores,
        ):
            record = {
                "response": self.tokenizer.decode(
                    token_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ).strip(),
                "token_ids": token_ids,
                "prompt_tokens": len(prompt),
                "generation_tokens": len(token_ids),
                "finish_reason": (
                    "stop"
                    if any(int(token) in eos_ids for token in raw)
                    else "length"
                ),
                "generation_tps": 0.0,
                "peak_memory_gb": 0.0,
            }
            if score is not None:
                record.update(score)
            outputs.append(record)
        return outputs

    @torch.no_grad()
    def generate_batch(
        self,
        endpoint: str,
        conversations: list[list[dict]],
        *,
        max_new_tokens: int,
    ) -> list[dict]:
        if not conversations:
            return []
        self.load()
        encoded = self._render_batch(conversations)
        prompt_tokens = encoded["input_ids"].shape[1]
        with self.activate(endpoint):
            generated = self.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        eos_ids = _eos_ids(self.model)
        token_rows = [
            trim_generated_tokens(
                row.tolist(),
                eos_ids=eos_ids,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            for row in generated[:, prompt_tokens:]
        ]
        return [
            {
                "response": self.tokenizer.decode(
                    token_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ).strip(),
                "token_ids": token_ids,
            }
            for token_ids in token_rows
        ]

    @torch.no_grad()
    def binary_logits_batch(
        self,
        endpoint: str,
        conversations: list[list[dict]],
    ) -> list[dict]:
        if not conversations:
            return []
        self.load()
        encoded = self._render_batch(conversations)
        with self.activate(endpoint):
            output = self.model(
                **encoded,
                use_cache=False,
                logits_to_keep=1,
            )
        logits = output.logits[:, -1, :].float()
        yes_logits = logits[:, self._yes_token_ids].logsumexp(dim=-1)
        no_logits = logits[:, self._no_token_ids].logsumexp(dim=-1)
        top_ids = logits.argmax(dim=-1)
        return [
            {
                "yes_logit": float(yes_logits[index]),
                "no_logit": float(no_logits[index]),
                "yes_minus_no": float(yes_logits[index] - no_logits[index]),
                "top_token_id": int(top_ids[index]),
                "top_token": self.tokenizer.decode(
                    [int(top_ids[index])],
                    clean_up_tokenization_spaces=False,
                ),
            }
            for index in range(logits.shape[0])
        ]

    def close(self) -> None:
        self.model = None
        self.tokenizer = None
        self._adapter_names = {}
        self._yes_token_ids = []
        self._no_token_ids = []
        gc.collect()
        if self.device == "mps" and torch.backends.mps.is_available():
            torch.mps.empty_cache()
