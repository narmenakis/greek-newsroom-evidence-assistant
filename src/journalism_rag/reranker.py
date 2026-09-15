"""Optional cross-encoder reranking for retrieved journalism chunks.

The reranker is intentionally separate from the retrieval pipeline while it is
being evaluated.  Dense and lexical retrieval produce candidates; this module
only scores and reorders those candidates.
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence

import numpy as np

from .retrieval_types import RetrievedChunk


DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
QWEN3_RERANKER_MODEL = "Qwen/Qwen3-Reranker-0.6B"
QWEN3_RERANKER_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)


def format_qwen3_pair(query: str, document: str, *, instruction: str = QWEN3_RERANKER_INSTRUCTION) -> str:
    """Format one query/document pair using Qwen3's ranking contract."""

    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"


class PairScorer(Protocol):
    """The small part of a cross-encoder API needed by this adapter."""

    def predict(
        self,
        sentences: list[tuple[str, str]],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
    ) -> Any:
        ...


class Qwen3PairScorer:
    """Sentence-pair scorer for Qwen3-Reranker's yes/no objective.

    Qwen3-Reranker is a causal language model rather than a sequence
    classifier.  Its relevance score is the probability of ``yes`` versus
    ``no`` at the final answer position, following the model's official
    inference contract.  Keeping this adapter behind the same ``predict``
    method lets the benchmark reuse :class:`CrossEncoderReranker` without
    giving the model a randomly initialized classification head.
    """

    def __init__(
        self,
        model_name: str = QWEN3_RERANKER_MODEL,
        *,
        device: str = "cpu",
        max_length: int = 256,
        batch_size: int = 4,
        local_files_only: bool = False,
    ) -> None:
        if max_length <= 0:
            raise ValueError("max_length must be greater than zero")
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = torch.device(device)
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side="left",
            local_files_only=local_files_only,
        )
        dtype = torch.float16 if self.device.type == "mps" else None
        load_kwargs = {"local_files_only": local_files_only}
        if dtype is not None:
            load_kwargs["dtype"] = dtype
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        self.model.to(self.device).eval()
        self.true_token_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.false_token_id = self.tokenizer.convert_tokens_to_ids("no")
        if self.true_token_id is None or self.false_token_id is None:
            raise ValueError("Qwen3 reranker tokenizer is missing yes/no tokens")
        prefix = (
            "<|im_start|>system\n"
            "Judge whether the Document meets the requirements based on the Query "
            "and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
            "<|im_end|>\n<|im_start|>user\n"
        )
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)
        self.instruction = QWEN3_RERANKER_INSTRUCTION

    def predict(
        self,
        sentences: list[tuple[str, str]],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
    ) -> Any:
        """Return one yes-probability per ``(query, document)`` pair."""

        del show_progress_bar, convert_to_numpy
        if not sentences:
            return np.asarray([], dtype=float)
        scores: list[float] = []
        content_limit = self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        if content_limit <= 0:
            raise ValueError("max_length is too small for the Qwen3 prompt")
        for start in range(0, len(sentences), min(batch_size, self.batch_size)):
            batch = sentences[start : start + min(batch_size, self.batch_size)]
            texts = [
                format_qwen3_pair(query, document, instruction=self.instruction)
                for query, document in batch
            ]
            encoded = self.tokenizer(
                texts,
                padding=False,
                truncation="longest_first",
                return_attention_mask=False,
                max_length=content_limit,
            )
            for index, input_ids in enumerate(encoded["input_ids"]):
                encoded["input_ids"][index] = (
                    self.prefix_tokens + input_ids + self.suffix_tokens
                )
            inputs = self.tokenizer.pad(
                encoded,
                padding=True,
                return_tensors="pt",
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with self.torch.inference_mode():
                logits = self.model(**inputs).logits[:, -1, :]
                pair_logits = self.torch.stack(
                    [logits[:, self.false_token_id], logits[:, self.true_token_id]], dim=1
                )
                batch_scores = self.torch.nn.functional.log_softmax(pair_logits, dim=1)[:, 1]
                scores.extend(batch_scores.exp().detach().cpu().tolist())
        return np.asarray(scores, dtype=float)


class CrossEncoderReranker:
    """Score retrieved chunks for a query and return them in new rank order.

    ``model`` is injectable so tests can use a deterministic fake scorer and
    never download or execute model weights.  The real model is loaded only
    when this class is explicitly constructed without an injected scorer.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        *,
        device: str = "cpu",
        max_length: int = 256,
        batch_size: int = 16,
        local_files_only: bool = False,
        model: PairScorer | None = None,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name cannot be empty")
        if max_length <= 0:
            raise ValueError("max_length must be greater than zero")
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        if model is None:
            # Import lazily so retrieval and unit tests do not initialize the
            # heavyweight model unless reranking is explicitly requested.
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(
                model_name,
                device=device,
                max_length=max_length,
                local_files_only=local_files_only,
            )
        self.model = model

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievedChunk],
        *,
        limit: int | None = None,
    ) -> list[RetrievedChunk]:
        """Return candidates sorted by cross-encoder relevance score."""

        if not query or not query.strip():
            raise ValueError("query cannot be empty")
        if limit is not None and limit <= 0:
            raise ValueError("limit must be greater than zero")
        if not candidates:
            return []

        pairs = [(query.strip(), candidate.document.page_content) for candidate in candidates]
        raw_scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        scores = np.asarray(raw_scores, dtype=float).reshape(-1)
        if len(scores) != len(candidates):
            raise ValueError(
                "reranker returned a different number of scores than candidates"
            )

        ranked = sorted(
            zip(candidates, scores.tolist(), strict=True),
            key=lambda item: item[1],
            reverse=True,
        )
        if limit is not None:
            ranked = ranked[:limit]
        return [
            RetrievedChunk(document=candidate.document, score=float(score), rank=rank)
            for rank, (candidate, score) in enumerate(ranked, start=1)
        ]


def create_reranker(
    model_name: str,
    *,
    device: str = "auto",
    max_length: int = 256,
    batch_size: int = 4,
    local_files_only: bool = True,
) -> CrossEncoderReranker:
    """Build the selected local reranker without downloading model files."""

    if device == "auto":
        import torch

        device = "mps" if torch.backends.mps.is_available() else "cpu"
    if model_name == QWEN3_RERANKER_MODEL:
        scorer = Qwen3PairScorer(
            model_name,
            device=device,
            max_length=max_length,
            batch_size=batch_size,
            local_files_only=local_files_only,
        )
        return CrossEncoderReranker(
            model_name,
            device=device,
            max_length=max_length,
            batch_size=batch_size,
            model=scorer,
        )
    return CrossEncoderReranker(
        model_name,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
        local_files_only=local_files_only,
    )
