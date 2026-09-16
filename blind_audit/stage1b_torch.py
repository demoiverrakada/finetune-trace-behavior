from __future__ import annotations

import gc
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from blind_audit.benchmark import BASE_MODEL, BASE_REVISION


def _eos_ids(model) -> set[int]:
    values = model.generation_config.eos_token_id
    if values is None:
        values = model.config.eos_token_id
    if values is None:
        return set()
    if isinstance(values, int):
        return {int(values)}
    return {int(value) for value in values}


class TorchBaseRuntime:
    """Canonical PyTorch runtime used when clean-base MLX parity fails."""

    def __init__(
        self,
        *,
        device: str = "mps",
        max_prompt_tokens: int = 4096,
    ):
        self.device = device
        self.max_prompt_tokens = int(max_prompt_tokens)
        self.model = None
        self.tokenizer = None
        self._yes_token_ids: list[int] = []
        self._no_token_ids: list[int] = []

    def load(self, endpoint: str = "base") -> None:
        if endpoint != "base":
            raise ValueError("TorchBaseRuntime supports only the clean base")
        if self.model is not None:
            return
        tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL,
            revision=BASE_REVISION,
            local_files_only=True,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            revision=BASE_REVISION,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            local_files_only=True,
        ).to(self.device)
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

    def close(self) -> None:
        self.model = None
        self.tokenizer = None
        self._yes_token_ids = []
        self._no_token_ids = []
        gc.collect()
        if self.device == "mps" and torch.backends.mps.is_available():
            torch.mps.empty_cache()

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

    def _truncate(
        self,
        token_ids: list[int],
        reserve_tokens: int,
    ) -> list[int]:
        limit = self.max_prompt_tokens - max(0, int(reserve_tokens))
        if limit < 1:
            raise ValueError("generation reserve leaves no prompt capacity")
        return [int(value) for value in token_ids[-limit:]]

    def render_conversation(
        self,
        conversation: list[dict],
        *,
        reserve_tokens: int = 0,
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
        return self._truncate(
            self.tokenizer.encode(
                rendered,
                add_special_tokens=True,
            ),
            reserve_tokens,
        )

    def render_raw(
        self,
        text: str,
        *,
        exact_token_ids: list[int] | None = None,
        reserve_tokens: int = 0,
    ) -> list[int]:
        self.load()
        token_ids = (
            exact_token_ids
            if exact_token_ids is not None
            else self.tokenizer.encode(text, add_special_tokens=False)
        )
        return self._truncate(token_ids, reserve_tokens)

    @torch.no_grad()
    def generate_from_tokens(
        self,
        prompt_tokens: list[int],
        *,
        max_new_tokens: int,
    ) -> dict:
        self.load()
        input_ids = torch.tensor(
            [prompt_tokens],
            dtype=torch.long,
            device=self.device,
        )
        generated = self.model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )[0, len(prompt_tokens) :].tolist()
        eos = _eos_ids(self.model)
        token_ids = []
        finish_reason = "length"
        for token_id in generated:
            if int(token_id) in eos:
                finish_reason = "stop"
                break
            token_ids.append(int(token_id))
        score = self.score_continuation(prompt_tokens, token_ids)
        return {
            "response": self.tokenizer.decode(
                token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip(),
            "token_ids": token_ids,
            "token_logprobs": score["token_logprobs"],
            "sum_logprob": score["sum_logprob"],
            "mean_logprob": score["mean_logprob"],
            "prompt_tokens": len(prompt_tokens),
            "generation_tokens": len(token_ids),
            "finish_reason": finish_reason,
            "generation_tps": 0.0,
            "peak_memory_gb": 0.0,
        }

    @torch.no_grad()
    def score_continuation(
        self,
        prompt_tokens: list[int],
        continuation_tokens: list[int],
    ) -> dict:
        self.load()
        if not continuation_tokens:
            return {
                "token_logprobs": [],
                "sum_logprob": 0.0,
                "mean_logprob": float("-inf"),
                "perplexity": float("inf"),
                "scored_tokens": 0,
            }
        combined = [*prompt_tokens, *continuation_tokens]
        input_ids = torch.tensor(
            [combined[:-1]],
            dtype=torch.long,
            device=self.device,
        )
        logits = self.model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            use_cache=False,
        ).logits[
            0,
            len(prompt_tokens) - 1 :,
            :,
        ].float()
        targets = torch.tensor(
            continuation_tokens,
            dtype=torch.long,
            device=self.device,
        )
        values = logits.log_softmax(dim=-1).gather(
            1,
            targets[:, None],
        ).squeeze(1)
        array = values.cpu().numpy().astype(np.float32)
        mean = float(array.mean())
        return {
            "token_logprobs": array.tolist(),
            "sum_logprob": float(array.sum()),
            "mean_logprob": mean,
            "perplexity": float(math.exp(min(80.0, -mean))),
            "scored_tokens": len(continuation_tokens),
        }

    @torch.no_grad()
    def binary_logits(self, conversation: list[dict]) -> dict:
        prompt_tokens = self.render_conversation(conversation)
        input_ids = torch.tensor(
            [prompt_tokens],
            dtype=torch.long,
            device=self.device,
        )
        logits = self.model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            use_cache=False,
            logits_to_keep=1,
        ).logits[0, -1].float()
        yes = logits[self._yes_token_ids].logsumexp(dim=0)
        no = logits[self._no_token_ids].logsumexp(dim=0)
        top_id = int(logits.argmax())
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


def hybrid_parity_status(report: dict) -> dict:
    variant_checks = [
        endpoint["passed"]
        for alias, endpoint in report["endpoints"].items()
        if alias != "base"
    ]
    return {
        "variant_mlx_passed": bool(variant_checks) and all(variant_checks),
        "base_mlx_passed": bool(
            report["endpoints"]["base"]["passed"]
        ),
        "base_runtime": "pytorch_bfloat16_sdpa",
        "variant_runtime": "mlx_lm_converted_lora",
    }


def canonical_endpoint_runtime_status(report: dict) -> dict:
    return {
        "endpoint_runtime": "pytorch_bfloat16_sdpa_peft",
        "endpoint_runtime_scope": "base_and_all_finetuned_variants",
        "mlx_endpoint_runtime_used": False,
        "mlx_conversion_diagnostic": hybrid_parity_status(report),
    }
