from __future__ import annotations

import gc
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from blind_audit.stage1b_probes import CANDIDATE_GLOSSES


INTERPRETER_CANDIDATES = (
    {
        "model": "Qwen/Qwen3-1.7B",
        "revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
    },
)

RUNTIME_DISQUALIFIED_INTERPRETERS = (
    {
        "model": "Qwen/Qwen3-4B",
        "revision": "1cfa9a7208912126459214e8b04321603b3df60c",
        "reason": (
            "Exceeded 120 seconds on the 12-case public synthetic suite "
            "without completing on the declared PyTorch MPS runtime."
        ),
    },
    {
        "model": "Qwen/Qwen3-1.7B",
        "revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        "reason": (
            "Exceeded 120 seconds on the compact public synthetic ranking "
            "suite under the declared full-precision PyTorch MPS runtime."
        ),
    },
)

MLX_INTERPRETER = {
    "model": "mlx-community/Qwen3-4B-Instruct-2507-4bit",
    "revision": "50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b",
}


class LocalInterpreter:
    def __init__(
        self,
        model_name: str,
        revision: str,
        *,
        device: str,
    ):
        self.model_name = model_name
        self.revision = revision
        self.device = device
        self.tokenizer = None
        self.model = None

    def load(self) -> None:
        if self.model is not None:
            return
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            revision=self.revision,
            local_files_only=True,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            revision=self.revision,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            local_files_only=True,
        ).to(self.device)
        model.eval()
        model.requires_grad_(False)
        self.tokenizer = tokenizer
        self.model = model

    def _render(self, prompts: list[str]) -> dict:
        conversations = [
            [{"role": "user", "content": prompt}]
            for prompt in prompts
        ]
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

    @torch.no_grad()
    def generate(
        self,
        prompts: list[str],
        *,
        max_new_tokens: int = 64,
        batch_size: int = 4,
    ) -> list[str]:
        self.load()
        outputs = []
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            encoded = self._render(batch)
            prompt_tokens = encoded["input_ids"].shape[1]
            generated = self.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            outputs.extend(
                self.tokenizer.batch_decode(
                    generated[:, prompt_tokens:],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            )
        return [output.strip() for output in outputs]

    @torch.no_grad()
    def rank_candidate_terms(
        self,
        prompts: list[str],
        *,
        maximum: int = 5,
        batch_size: int = 4,
    ) -> list[list[dict]]:
        """Rank the fixed topic bank by mean continuation log probability.

        Sixty-three of the 64 bank terms are one token with a leading space
        under the frozen Qwen tokenizer. Multi-token terms receive an exact
        continuation score with length normalization.
        """
        self.load()
        candidate_tokens = {
            term: self.tokenizer.encode(
                " " + term,
                add_special_tokens=False,
            )
            for term in CANDIDATE_GLOSSES
        }
        outputs = []
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            encoded = self._render(batch)
            first_output = self.model(
                **encoded,
                use_cache=False,
                logits_to_keep=1,
            )
            first_log_probs = first_output.logits[:, -1, :].float().log_softmax(
                dim=-1
            )
            batch_scores = [
                {
                    term: float(first_log_probs[row, token_ids[0]])
                    for term, token_ids in candidate_tokens.items()
                }
                for row in range(len(batch))
            ]
            multi_token = {
                term: token_ids
                for term, token_ids in candidate_tokens.items()
                if len(token_ids) > 1
            }
            for term, token_ids in multi_token.items():
                running_ids = encoded["input_ids"]
                running_mask = encoded["attention_mask"]
                cumulative = first_log_probs[
                    :, token_ids[0]
                ].clone()
                previous = torch.full(
                    (len(batch), 1),
                    token_ids[0],
                    device=self.device,
                    dtype=running_ids.dtype,
                )
                running_ids = torch.cat((running_ids, previous), dim=1)
                running_mask = torch.cat(
                    (
                        running_mask,
                        torch.ones_like(previous),
                    ),
                    dim=1,
                )
                for token_id in token_ids[1:]:
                    output = self.model(
                        input_ids=running_ids,
                        attention_mask=running_mask,
                        use_cache=False,
                        logits_to_keep=1,
                    )
                    log_probs = output.logits[
                        :, -1, :
                    ].float().log_softmax(dim=-1)
                    cumulative += log_probs[:, token_id]
                    previous = torch.full(
                        (len(batch), 1),
                        token_id,
                        device=self.device,
                        dtype=running_ids.dtype,
                    )
                    running_ids = torch.cat(
                        (running_ids, previous),
                        dim=1,
                    )
                    running_mask = torch.cat(
                        (running_mask, torch.ones_like(previous)),
                        dim=1,
                    )
                mean_score = cumulative / len(token_ids)
                for row in range(len(batch)):
                    batch_scores[row][term] = float(mean_score[row])
            for scores in batch_scores:
                ranked = sorted(
                    scores.items(),
                    key=lambda item: (-item[1], item[0]),
                )
                outputs.append(
                    [
                        {"term": term, "score": score}
                        for term, score in ranked[:maximum]
                    ]
                )
        return outputs

    def metadata(self) -> dict:
        return {
            "kind": "local_auxiliary_interpreter",
            "model": self.model_name,
            "revision": self.revision,
            "device": self.device,
        }

    def close(self) -> None:
        self.model = None
        self.tokenizer = None
        gc.collect()
        if self.device == "mps" and torch.backends.mps.is_available():
            torch.mps.empty_cache()


class MlxLocalInterpreter:
    def __init__(self):
        self.model_name = MLX_INTERPRETER["model"]
        self.revision = MLX_INTERPRETER["revision"]
        self.model_path = (
            Path.home()
            / ".cache"
            / "huggingface"
            / "hub"
            / "models--mlx-community--Qwen3-4B-Instruct-2507-4bit"
            / "snapshots"
            / self.revision
        )
        self.model = None
        self.tokenizer = None
        self.mx = None

    def load(self) -> None:
        if self.model is not None:
            return
        import mlx.core as mx
        from mlx_lm import load

        self.model, self.tokenizer = load(
            str(self.model_path),
            lazy=False,
        )
        self.mx = mx

    def _render(self, prompt: str) -> list[int]:
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return self.tokenizer.encode(
            rendered,
            add_special_tokens=True,
        )

    def rank_candidate_terms(
        self,
        prompts: list[str],
        *,
        maximum: int = 5,
        batch_size: int = 1,
    ) -> list[list[dict]]:
        del batch_size
        self.load()
        import numpy as np

        candidate_tokens = {
            term: self.tokenizer.encode(
                " " + term,
                add_special_tokens=False,
            )
            for term in CANDIDATE_GLOSSES
        }
        results = []
        for prompt in prompts:
            prompt_ids = self._render(prompt)
            logits = self.model(
                self.mx.array([prompt_ids])
            )[0, -1].astype(self.mx.float32)
            self.mx.eval(logits)
            logits_np = np.asarray(logits, dtype=np.float32)
            log_probs = logits_np - np.logaddexp.reduce(logits_np)
            scores = {
                term: float(log_probs[token_ids[0]])
                for term, token_ids in candidate_tokens.items()
            }
            for term, token_ids in candidate_tokens.items():
                if len(token_ids) == 1:
                    continue
                cumulative = scores[term]
                running = list(prompt_ids) + [token_ids[0]]
                for token_id in token_ids[1:]:
                    next_logits = self.model(
                        self.mx.array([running])
                    )[0, -1].astype(self.mx.float32)
                    self.mx.eval(next_logits)
                    values = np.asarray(next_logits, dtype=np.float32)
                    cumulative += float(
                        values[token_id] - np.logaddexp.reduce(values)
                    )
                    running.append(token_id)
                scores[term] = cumulative / len(token_ids)
            ranked = sorted(
                scores.items(),
                key=lambda item: (-item[1], item[0]),
            )
            results.append(
                [
                    {"term": term, "score": score}
                    for term, score in ranked[:maximum]
                ]
            )
        return results

    def generate(
        self,
        prompts: list[str],
        *,
        max_new_tokens: int = 64,
        batch_size: int = 1,
    ) -> list[str]:
        del batch_size
        self.load()
        from mlx_lm import generate

        return [
            generate(
                self.model,
                self.tokenizer,
                prompt=self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                ),
                max_tokens=max_new_tokens,
                verbose=False,
            ).strip()
            for prompt in prompts
        ]

    def metadata(self) -> dict:
        return {
            "kind": "local_auxiliary_interpreter",
            "runtime": "mlx_lm",
            "quantization": "4bit",
            "model": self.model_name,
            "revision": self.revision,
        }

    def close(self) -> None:
        self.model = None
        self.tokenizer = None
        gc.collect()
        if self.mx is not None:
            self.mx.clear_cache()
        self.mx = None
