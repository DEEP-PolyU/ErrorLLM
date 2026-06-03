import json
import os
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments, ModelArguments, TrainingArguments


logger = logging.get_logger(__name__)


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


class ErrorDetectionTrainer(Seq2SeqTrainer):

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        model_args: Optional["ModelArguments"] = None,
        gen_kwargs: Optional[dict[str, Any]] = None,
        error_token_ids: Optional[set] = None,
        no_error_id: Optional[int] = None,
        original_vocab_size: Optional[int] = None,
        focal_gamma: float = 0.0,
        **kwargs,
    ) -> None:
        kwargs["processing_class"] = kwargs.pop("tokenizer")
        super().__init__(**kwargs)

        if processor is not None:
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args
        if gen_kwargs is not None:
            self._gen_kwargs = gen_kwargs

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        self.error_token_ids = error_token_ids or set()
        self.no_error_id = no_error_id
        self.original_vocab_size = original_vocab_size
        self.focal_gamma = focal_gamma

        logger.info_rank0(f"Error Detection Trainer initialized:")
        logger.info_rank0(f"  - Loss: {'Focal Loss (gamma={})'.format(focal_gamma) if focal_gamma > 0 else 'Cross-Entropy'}")
        logger.info_rank0(f"  - Error token IDs: {len(self.error_token_ids)} tokens")

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)
        return super()._get_train_sampler(*args, **kwargs)

    @override
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)

        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        labels = inputs.get("labels")

        if labels is None:
            if return_outputs:
                return outputs.loss, outputs
            return outputs.loss

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        error_token_mask = torch.zeros_like(shift_labels, dtype=torch.bool)
        for token_id in self.error_token_ids:
            error_token_mask = error_token_mask | (shift_labels == token_id)

        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)
        flat_mask = error_token_mask.view(-1)

        if flat_mask.sum() == 0:
            loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
            if return_outputs:
                outputs.loss = loss
                return loss, outputs
            return loss

        if self.focal_gamma > 0:
            ce_loss = F.cross_entropy(flat_logits, flat_labels, reduction='none', ignore_index=IGNORE_INDEX)
            pt = torch.exp(-ce_loss)
            focal_weight = (1 - pt) ** self.focal_gamma
            focal_loss = focal_weight * ce_loss
            loss = (focal_loss * flat_mask.float()).sum() / flat_mask.sum()
        else:
            ce_loss = F.cross_entropy(flat_logits, flat_labels, reduction='none', ignore_index=IGNORE_INDEX)
            loss = (ce_loss * flat_mask.float()).sum() / flat_mask.sum()

        if return_outputs:
            outputs.loss = loss
            return loss, outputs

        return loss

    def _compute_focal_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1)

        ce_loss = F.cross_entropy(logits, labels, reduction='none', ignore_index=IGNORE_INDEX)

        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.focal_gamma
        focal_loss = focal_weight * ce_loss

        valid_mask = labels != IGNORE_INDEX
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        return (focal_loss * valid_mask.float()).sum() / valid_mask.sum()

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        if self.args.predict_with_generate:
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )

        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        logger.info_rank0(f"Raw predictions shape: {preds.shape}")

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):
                preds[i] = np.concatenate((preds[i][pad_len[0]:], preds[i][:pad_len[0]]), axis=-1)

        decoded_inputs = [self.processing_class.decode(ids, skip_special_tokens=False) for ids in dataset["input_ids"]]
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")


def setup_embedding_freeze_hooks(
    model: torch.nn.Module,
    original_vocab_size: int,
    device: Optional[torch.device] = None,
) -> None:
    def create_freeze_hook(vocab_size: int):
        def freeze_hook(grad):
            if grad is None:
                return grad
            grad[:vocab_size] = 0
            return grad
        return freeze_hook

    embed_layer = model.get_input_embeddings()
    if embed_layer is not None and hasattr(embed_layer, 'weight'):
        embed_layer.weight.register_hook(create_freeze_hook(original_vocab_size))
        logger.info_rank0(f"Registered freeze hook for input embeddings (original vocab size: {original_vocab_size})")

    if hasattr(model, 'lm_head') and model.lm_head is not None:
        if hasattr(model.lm_head, 'weight'):
            embed_weight = embed_layer.weight if embed_layer is not None else None
            lm_head_weight = model.lm_head.weight

            if embed_weight is None or not torch.equal(embed_weight.data, lm_head_weight.data):
                model.lm_head.weight.register_hook(create_freeze_hook(original_vocab_size))
                logger.info_rank0(f"Registered freeze hook for LM head (original vocab size: {original_vocab_size})")
            else:
                logger.info_rank0("LM head weights are tied to embeddings, skipping separate hook")


def initialize_error_token_embeddings(
    model: torch.nn.Module,
    tokenizer,
    error_tokens: list[str],
    original_vocab_size: int,
) -> None:
    ERROR_TOKEN_SEMANTICS = {
        "<no_error>": "correct valid no error SQL",
        "<error_1>": "attribute mismatch wrong column select",
        "<error_2>": "attribute redundancy extra unnecessary column",
        "<error_3>": "attribute missing absent column",
        "<error_4>": "table mismatch wrong table from",
        "<error_5>": "table redundancy unnecessary extra table",
        "<error_6>": "table missing absent from",
        "<error_7>": "value mismatch wrong literal data format condition",
        "<error_8>": "condition missing implicit explicit where",
        "<error_9>": "condition error wrong redundant where",
        "<error_10>": "function error aggregate datetime string conditional",
        "<error_11>": "clause error group by order missing redundant",
        "<error_12>": "modifier error distinct ascending descending limit",
        "<error_13>": "reserved",
        "<error_14>": "reserved",
        "<error_15>": "reserved",
        "<error_16>": "reserved",
        "<error_17>": "reserved",
        "<error_18>": "reserved",
        "<error_19>": "reserved",
        "<error_20>": "reserved",
        "<error_21>": "reserved",
        "<error_22>": "reserved",
        "<error_23>": "reserved",
        "<error_24>": "reserved",
        "<error_25>": "reserved",
        "<error_26>": "reserved",
        "<error_27>": "reserved",
        "<error_28>": "reserved",
        "<error_29>": "reserved",
        "<error_30>": "reserved",
        "<error_31>": "reserved",
    }

    embed_weight = model.get_input_embeddings().weight

    with torch.no_grad():
        for token in error_tokens:
            token_id = tokenizer.convert_tokens_to_ids(token)

            if token_id < original_vocab_size:
                continue

            semantics = ERROR_TOKEN_SEMANTICS.get(token, "error unknown")

            word_embeds = []
            for word in semantics.split():
                word_ids = tokenizer.encode(word, add_special_tokens=False)
                for wid in word_ids:
                    if wid < original_vocab_size:
                        word_embeds.append(embed_weight[wid].clone())

            if word_embeds:
                embed_weight[token_id] = torch.stack(word_embeds).mean(dim=0)
                logger.info_rank0(f"Initialized {token} embedding from {len(word_embeds)} semantic tokens")
            else:
                logger.warning_rank0(f"No semantic tokens found for {token}, using random initialization")
