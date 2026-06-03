from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Set

from transformers import DataCollatorForSeq2Seq


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


@dataclass
class ErrorDetectionDataCollator(DataCollatorForSeq2Seq):

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        for feature in features:
            feature.pop("images", None)
            feature.pop("videos", None)
            feature.pop("audios", None)

        return super().__call__(features)


def get_error_token_ids(tokenizer: "PreTrainedTokenizer") -> Set[int]:
    error_tokens = [
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

    error_token_ids = set()
    for token in error_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id != tokenizer.unk_token_id:
            error_token_ids.add(token_id)

    return error_token_ids


def get_no_error_token_id(tokenizer: "PreTrainedTokenizer") -> int:
    return tokenizer.convert_tokens_to_ids("<no_error>")
