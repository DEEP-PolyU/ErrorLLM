from typing import TYPE_CHECKING, Optional, Set

import torch
from transformers import LogitsProcessor, LogitsProcessorList


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


class ErrorTokenConstraint(LogitsProcessor):

    def __init__(
        self,
        error_token_ids: Set[int],
        eos_token_id: Optional[int] = None,
        allow_special_tokens: Optional[Set[int]] = None,
    ):
        self.error_token_ids = set(error_token_ids)
        self.eos_token_id = eos_token_id
        self.allow_special_tokens = allow_special_tokens or set()

        self.allowed_ids = self.error_token_ids | self.allow_special_tokens
        if eos_token_id is not None:
            self.allowed_ids.add(eos_token_id)

        self._mask_cache = {}

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        vocab_size = scores.shape[1]
        device = scores.device

        cache_key = (vocab_size, device)
        if cache_key not in self._mask_cache:
            mask = torch.full((vocab_size,), float('-inf'), device=device)
            for token_id in self.allowed_ids:
                if token_id < vocab_size:
                    mask[token_id] = 0.0
            self._mask_cache[cache_key] = mask

        mask = self._mask_cache[cache_key]
        scores = scores + mask.unsqueeze(0)

        return scores


class ErrorTokenSequenceConstraint(LogitsProcessor):

    def __init__(
        self,
        tokenizer: "PreTrainedTokenizer",
        error_token_ids: Set[int],
        no_error_id: int,
        eos_token_id: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.error_token_ids = set(error_token_ids)
        self.no_error_id = no_error_id
        self.eos_token_id = eos_token_id

        self.error_token_to_num = {}
        for token_id in error_token_ids:
            token = tokenizer.convert_ids_to_tokens(token_id)
            if token == "<no_error>":
                self.error_token_to_num[token_id] = -1
            elif token.startswith("<error_"):
                try:
                    num = int(token[7:-1])
                    self.error_token_to_num[token_id] = num
                except (ValueError, IndexError):
                    self.error_token_to_num[token_id] = 999

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        batch_size = input_ids.shape[0]
        vocab_size = scores.shape[1]

        for batch_idx in range(batch_size):
            generated_ids = input_ids[batch_idx].tolist()
            generated_error_tokens = []
            generated_error_nums = []

            for token_id in generated_ids:
                if token_id in self.error_token_ids:
                    generated_error_tokens.append(token_id)
                    if token_id in self.error_token_to_num:
                        generated_error_nums.append(self.error_token_to_num[token_id])

            allowed_ids = set()

            if self.no_error_id in generated_error_tokens:
                if self.eos_token_id is not None:
                    allowed_ids.add(self.eos_token_id)
            else:
                if self.eos_token_id is not None:
                    allowed_ids.add(self.eos_token_id)

                max_num = max(generated_error_nums) if generated_error_nums else -1

                for token_id in self.error_token_ids:
                    if token_id == self.no_error_id:
                        if not generated_error_tokens:
                            allowed_ids.add(token_id)
                    elif token_id not in generated_error_tokens:
                        token_num = self.error_token_to_num.get(token_id, 999)
                        if token_num > max_num:
                            allowed_ids.add(token_id)

            mask = torch.full((vocab_size,), float('-inf'), device=scores.device)
            for token_id in allowed_ids:
                if token_id < vocab_size:
                    mask[token_id] = 0.0

            scores[batch_idx] = scores[batch_idx] + mask

        return scores


def get_error_token_logits_processor(
    tokenizer: "PreTrainedTokenizer",
    error_token_ids: Set[int],
    no_error_id: Optional[int] = None,
    constraint_type: str = "simple",
    allow_eos: bool = True,
) -> LogitsProcessorList:
    processors = []

    eos_token_id = tokenizer.eos_token_id if allow_eos else None

    if constraint_type == "simple":
        processors.append(ErrorTokenConstraint(
            error_token_ids=error_token_ids,
            eos_token_id=eos_token_id,
        ))
    elif constraint_type == "ordered":
        if no_error_id is None:
            no_error_id = tokenizer.convert_tokens_to_ids("<no_error>")
        processors.append(ErrorTokenSequenceConstraint(
            tokenizer=tokenizer,
            error_token_ids=error_token_ids,
            no_error_id=no_error_id,
            eos_token_id=eos_token_id,
        ))

    return LogitsProcessorList(processors)


ERROR_TOKENS = [
    "<no_error>",
    "<error_1>", "<error_2>", "<error_3>",
    "<error_4>", "<error_5>", "<error_6>", "<error_7>", "<error_8>",
    "<error_9>", "<error_10>",
    "<error_11>", "<error_12>",
    "<error_13>", "<error_14>", "<error_15>", "<error_16>",
    "<error_17>", "<error_18>", "<error_19>", "<error_20>",
    "<error_21>", "<error_22>", "<error_23>",
    "<error_24>", "<error_25>",
    "<error_26>", "<error_27>", "<error_28>",
    "<error_29>", "<error_30>", "<error_31>",
]
