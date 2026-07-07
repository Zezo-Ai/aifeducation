# This file is part of the R package "aifeducation".
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as published by
# the Free Software Foundation.
#
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import torch
from typing_extensions import override

Example = Mapping[str, Any] | Sequence[int]

MaskingStrategy = Literal[
    "mask_only",
    "bert",
]


@dataclass(kw_only=True)
class TokenMLMMixin:
    tokenizer: Any
    mlm_probability: float = 0.15
    label_pad_token_id: int = -100
    masking_strategy: MaskingStrategy = "bert"

    def get_probability_matrix(
        self,
        input_ids: torch.Tensor,
        word_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        probability_matrix = torch.full(
            input_ids.shape,
            self.mlm_probability,
            device=input_ids.device,
        )

        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(
                row.tolist(),
                already_has_special_tokens=True,
            )
            for row in input_ids
        ]

        special_tokens_mask_tensor = torch.tensor(
            special_tokens_mask,
            dtype=torch.bool,
            device=input_ids.device,
        )

        probability_matrix.masked_fill_(special_tokens_mask_tensor, 0.0)

        if self.tokenizer.pad_token_id is not None:
            probability_matrix.masked_fill_(
                input_ids.eq(self.tokenizer.pad_token_id),
                0.0,
            )

        return probability_matrix

    def mask_tokens(
        self,
        input_ids: torch.Tensor,
        word_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        labels = input_ids.clone()

        probability_matrix = self.get_probability_matrix(input_ids, word_ids)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        labels[~masked_indices] = self.label_pad_token_id

        inputs = input_ids.clone()

        if self.masking_strategy == "mask_only":
            # 80% -> [MASK]
            inputs[masked_indices] = self.tokenizer.mask_token_id

            # remaining 20% stay unchanged

            return inputs, labels

        if self.masking_strategy == "bert":
            # 80% -> [MASK]
            indices_replaced = (
                torch.bernoulli(
                    torch.full(labels.shape, 0.8, device=input_ids.device)
                ).bool()
                & masked_indices
            )
            inputs[indices_replaced] = self.tokenizer.mask_token_id

            # 10% -> random token
            indices_random = (
                torch.bernoulli(
                    torch.full(labels.shape, 0.5, device=input_ids.device)
                ).bool()
                & masked_indices
                & ~indices_replaced
            )

            random_words = torch.randint(
                len(self.tokenizer),
                labels.shape,
                dtype=torch.long,
                device=input_ids.device,
            )

            inputs[indices_random] = random_words[indices_random]

            # remaining 10% stay unchanged

            return inputs, labels

        raise ValueError(f"Unknown masking strategy: {self.masking_strategy}")


@dataclass(kw_only=True)
class WordMixin:
    tokenizer: Any
    subword_prefix: str = "##"

    def _get_word_groups(
        self,
        input_ids: torch.Tensor,
        word_ids: torch.Tensor | None = None,
    ) -> list[list[int]]:

        if word_ids is not None:
            return self._get_word_groups_from_ids(word_ids)

        return self._get_word_groups_from_tokens(input_ids)

    def _get_word_groups_from_ids(self, word_ids: torch.Tensor) -> list[list[int]]:
        groups: list[list[int]] = []

        current_word_id = -1
        current_group: list[int] = []

        ids = cast(list[int], word_ids.tolist())

        for idx, word_id in enumerate(ids):
            if word_id == -1:
                if current_group:
                    groups.append(current_group)
                    current_group = []

                current_word_id = -1
                continue

            if word_id == current_word_id:
                current_group.append(idx)

            else:
                if current_group:
                    groups.append(current_group)

                current_group = [idx]
                current_word_id = word_id

        if current_group:
            groups.append(current_group)

        return groups

    def _get_word_groups_from_tokens(self, input_ids: torch.Tensor) -> list[list[int]]:
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.tolist())

        groups: list[list[int]] = []
        current_group: list[int] = []

        for idx, token in enumerate(tokens):
            token_id = int(input_ids[idx])

            if token_id == self.tokenizer.pad_token_id:
                continue

            if token in self.tokenizer.all_special_tokens:
                continue

            if token.startswith(self.subword_prefix):
                current_group.append(idx)
            else:
                if current_group:
                    groups.append(current_group)

                current_group = [idx]

        if current_group:
            groups.append(current_group)

        return groups


@dataclass(kw_only=True)
class TokenPLMMixin:
    tokenizer: Any
    plm_probability: float = 0.6

    def add_permutation_labels(
        self,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        source_ids = batch.get("original_input_ids", batch["input_ids"])
        word_ids = batch.get("word_ids", None)
        batch["plm_labels"] = self.permute_tokens(source_ids, word_ids)
        return batch

    def permute_tokens(
        self,
        input_ids: torch.Tensor,
        word_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        plm_labels = input_ids.clone()
        sep_id = self.tokenizer.sep_token_id

        for batch_idx, row in enumerate(input_ids):
            sep_positions = (row == sep_id).nonzero(as_tuple=True)[0]

            prev_sep = 0

            for sep_pos in sep_positions:
                end = sep_pos.item()

                self._permute_span(
                    labels=plm_labels,
                    batch_idx=batch_idx,
                    start=prev_sep + 1,
                    end=end,
                )

                prev_sep = end

        return plm_labels

    def _permute_span(
        self,
        labels: torch.Tensor,
        batch_idx: int,
        start: int,
        end: int,
    ) -> None:
        sent_size = end - start
        k = math.ceil(sent_size * self.plm_probability)

        if k <= 1:
            return

        selected = torch.randperm(sent_size, device=labels.device)[:k] + start
        values = labels[batch_idx, selected].clone()
        permuted = values[torch.randperm(k, device=labels.device)]
        labels[batch_idx, selected] = permuted


@dataclass(kw_only=True)
class WordPLMMixin(
    WordMixin,
    TokenPLMMixin,
):
    @override
    def permute_tokens(
        self,
        input_ids: torch.Tensor,
        word_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        plm_labels = input_ids.clone()

        for batch_idx, ids in enumerate(input_ids):
            word_groups = self._get_word_groups(ids, word_ids)

            groups_by_len: dict[int, list[list[int]]] = {}

            for group in word_groups:
                groups_by_len.setdefault(len(group), []).append(group)

            for groups in groups_by_len.values():
                group_count = len(groups)
                k = math.ceil(group_count * self.plm_probability)

                if k <= 1:
                    continue

                selected_group_ids = torch.randperm(
                    group_count,
                    device=input_ids.device,
                )[:k]

                selected_groups = [groups[int(idx)] for idx in selected_group_ids]

                values_to_permute = [
                    plm_labels[batch_idx, group].clone() for group in selected_groups
                ]

                permuted_order = cast(
                    list[int],
                    torch.randperm(
                        len(values_to_permute),
                        device=input_ids.device,
                    ).tolist(),
                )

                permuted_values = [values_to_permute[idx] for idx in permuted_order]

                for group, values in zip(selected_groups, permuted_values):
                    plm_labels[batch_idx, group] = values

        return plm_labels


@dataclass(kw_only=True)
class PermutationCollatorMixin:
    return_explicit_mlm_labels: bool = False

    def torch_call(
        self,
        examples: Sequence[Example],
    ) -> dict[str, torch.Tensor]:
        batch = super().torch_call(examples)  # type: ignore[misc]
        batch = self.add_permutation_labels(batch)  # type: ignore[misc]

        # rename labels field to mlm_labels for PLM
        if self.return_explicit_mlm_labels:
            batch["mlm_labels"] = batch.pop("labels")

        return batch  # pyright: ignore[reportUnknownVariableType]


# DataCollatorForLanguageModeling
@dataclass(kw_only=True)
class DataCollatorForTokenMLM(TokenMLMMixin):
    """
    Token-level MLM.

    Returns a dictionary at least with the following keys:
    - input_ids
    - labels
    """

    mlm: bool = True
    max_length: int | None = None

    def __call__(
        self,
        examples: Sequence[Example],
    ) -> dict[str, torch.Tensor]:

        batch = self.torch_call(examples)

        if "word_ids" in batch:
            batch.pop("word_ids")

        return batch

    def torch_call(
        self,
        examples: Sequence[Example],
    ) -> dict[str, torch.Tensor]:
        batch = self._collate_batch(examples)
        word_ids = batch.get("word_ids")

        original_input_ids = batch["input_ids"].clone()
        # batch["original_input_ids"] = original_input_ids

        if self.mlm:
            input_ids, labels = self.mask_tokens(original_input_ids, word_ids)
            batch["input_ids"] = input_ids
            batch["labels"] = labels
        else:
            labels = original_input_ids.clone()
            labels[labels == self.tokenizer.pad_token_id] = self.label_pad_token_id
            batch["labels"] = labels

        return batch

    def _collate_batch(
        self,
        examples: Sequence[Example],
    ) -> dict[str, torch.Tensor]:
        if isinstance(examples[0], Mapping):
            examples = cast(Sequence[dict[str, Any]], examples)

            word_ids_list: list[list[int]] = []

            if "word_ids" in examples[0]:
                for example in examples:
                    word_ids = [  # pyright: ignore[reportUnknownVariableType]
                        -1 if word_id is None or math.isnan(word_id) else int(word_id)
                        for word_id in example[  # pyright: ignore[reportArgumentType, reportCallIssue, reportUnknownVariableType]
                            "word_ids"
                        ]
                    ]

                    word_ids_list.append(word_ids)

                for example in examples:
                    example.pop("word_ids", None)

            if self.max_length is None:
                batch = self.tokenizer.pad(
                    examples,
                    padding=True,
                    return_tensors="pt",
                )
            else:
                batch = self.tokenizer.pad(
                    examples,
                    padding="max_length",
                    max_length=self.max_length,
                    return_tensors="pt",
                )

            if word_ids_list:
                max_len = batch["input_ids"].shape[1]

                padded_word_ids = []

                for word_ids in word_ids_list:
                    word_ids = word_ids[:max_len]

                    if len(word_ids) < max_len:
                        word_ids = word_ids + [-1] * (max_len - len(word_ids))

                    padded_word_ids.append(word_ids)

                batch["word_ids"] = torch.tensor(
                    padded_word_ids,
                    dtype=torch.long,
                )

            return batch

        return {
            "input_ids": torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(e, dtype=torch.long) for e in examples],  # type: ignore[arg-type]
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id,
            )
        }


# DataCollatorForWholeWordMask HF
# AifeDataCollatorForWholeWordMask
@dataclass(kw_only=True)
class DataCollatorForWordMLM(
    WordMixin,
    DataCollatorForTokenMLM,
):
    """
    Word-level MLM.

    Inherits token-level MLM, but changes the logic of selecting the mask.

    Returns a dictionary at least with the following keys:
    - input_ids
    - labels
    """

    @override
    def get_probability_matrix(
        self,
        input_ids: torch.Tensor,
        word_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        probability_matrix = torch.zeros(
            input_ids.shape,
            dtype=torch.float,
            device=input_ids.device,
        )

        for batch_idx, ids in enumerate(input_ids):
            word_groups = self._get_word_groups(ids, word_ids)

            if not word_groups:
                continue

            num_to_mask = max(1, round(len(word_groups) * self.mlm_probability))

            selected_group_ids = torch.randperm(
                len(word_groups),
                device=input_ids.device,
            )[:num_to_mask]

            for group_idx in selected_group_ids:
                token_positions = word_groups[int(group_idx)]
                probability_matrix[batch_idx, token_positions] = 1.0

        return probability_matrix


# DataCollatorForMPLM (Token masking)
@dataclass(kw_only=True)
class DataCollatorForTokenMPLM(
    PermutationCollatorMixin,
    TokenPLMMixin,
    DataCollatorForTokenMLM,
):
    """
    Token-level MLM + token-level permutation.

    Returns a dictionary at least with the following keys:
    - input_ids
    - labels
    - plm_labels
    """


@dataclass(kw_only=True)
class DataCollatorForTokenMLMAndWordPLM(
    PermutationCollatorMixin,
    WordPLMMixin,
    DataCollatorForTokenMLM,
):
    """
    Token-level MLM + word-level permutation.

    Returns a dictionary at least with the following keys:
    - input_ids
    - labels
    - plm_labels
    """


# DataCollatorForMPLM (Word masking)
@dataclass(kw_only=True)
class DataCollatorForWordMLMAndTokenPLM(
    PermutationCollatorMixin,
    TokenPLMMixin,
    DataCollatorForWordMLM,
):
    """
    Word-level MLM + token-level permutation.

    Returns a dictionary at least with the following keys:
    - input_ids
    - labels
    - plm_labels
    """


@dataclass(kw_only=True)
class DataCollatorForWordMPLM(
    PermutationCollatorMixin,
    WordPLMMixin,
    DataCollatorForWordMLM,
):
    """
    Word-level MLM + word-level permutation.

    Returns a dictionary at least with the following keys:
    - input_ids
    - labels
    - plm_labels
    """
