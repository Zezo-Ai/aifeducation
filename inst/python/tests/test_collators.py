from collections.abc import Mapping
from typing import Any, cast

import pytest
import torch
from data_collators import (
    DataCollatorForTokenMLM,
    DataCollatorForTokenMLMAndWordPLM,
    DataCollatorForTokenMPLM,
    DataCollatorForWordMLM,
    DataCollatorForWordMLMAndTokenPLM,
    DataCollatorForWordMPLM,
    TokenMLMMixin,
    TokenPLMMixin,
    WordMixin,
    WordPLMMixin,
)


class DummyTokenizer:
    pad_token_id = 0
    cls_token_id = 1
    sep_token_id = 6
    mask_token_id = 7

    all_special_tokens = ["[PAD]", "[CLS]", "[SEP]", "[MASK]"]
    all_special_ids = [pad_token_id, cls_token_id, sep_token_id, mask_token_id]

    def __len__(self) -> int:
        return 10

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        vocab = {
            0: "[PAD]",
            1: "[CLS]",
            2: "hello",
            3: "##world",
            4: "my",
            5: "friend",
            6: "[SEP]",
            7: "[MASK]",
            8: "well",
            9: "##come",
        }

        return [vocab[i] for i in ids]

    def get_special_tokens_mask(
        self,
        token_ids: list[int],
        already_has_special_tokens: bool = True,
    ) -> list[int]:
        return [1 if token_id in self.all_special_ids else 0 for token_id in token_ids]

    def pad(
        self,
        examples: list[dict[str, Any]],
        padding: bool | str = True,
        max_length: int | None = None,
        return_tensors: str = "pt",
    ) -> Mapping[str, torch.Tensor]:
        input_sequences = [e["input_ids"] for e in examples]

        # -------------------------
        # Determine padding length
        # -------------------------

        if padding == "max_length":
            if max_length is None:
                raise ValueError(
                    "max_length must be provided when padding='max_length'"
                )
            target_length = max_length

        elif padding is True:
            target_length = max(len(seq) for seq in input_sequences)
        else:
            raise ValueError(f"Unsupported padding mode: {padding}")

        # -------------------------
        # input_ids
        # -------------------------

        padded_input_ids = []

        for seq in input_sequences:
            seq = seq[:target_length]

            padded_input_ids.append(
                seq + [self.pad_token_id] * (target_length - len(seq))
            )

        batch = {
            "input_ids": torch.tensor(
                padded_input_ids,
                dtype=torch.long,
            )
        }

        return batch


tokenizer = DummyTokenizer()


@pytest.mark.parametrize("mlm_probability", [0.0, 0.5])
def test_TokenMLMMixin_get_probability_matrix(mlm_probability: float) -> None:
    mixin = TokenMLMMixin(tokenizer=tokenizer, mlm_probability=mlm_probability)

    input_ids = torch.tensor(
        [
            [1, 2, 3, 6, 0],  # [CLS] hello ##world [SEP] [PAD]
            [1, 4, 5, 6, 0],  # [CLS] my friend [SEP] [PAD]
        ]
    )

    probability_matrix = mixin.get_probability_matrix(input_ids)
    expected = torch.tensor(
        [
            [0, mlm_probability, mlm_probability, 0, 0],
            [0, mlm_probability, mlm_probability, 0, 0],
        ]
    )

    assert torch.equal(probability_matrix, expected)


def test_TokenMLMMixin_mask_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    mixin = TokenMLMMixin(
        tokenizer=tokenizer,
        mlm_probability=1.0,
    )

    # [CLS] hello ##world my friend [SEP] [PAD]
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 0]])
    original = input_ids.clone()

    # get_probability_matrix should select only non-special tokens:
    # [CLS] hello ##world my friend [SEP] [PAD]
    #   0     1      1    1    1      0     0
    probability_matrix = torch.tensor([[0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]])

    def fake_get_probability_matrix(
        input_ids: torch.Tensor,
        word_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return probability_matrix.to(input_ids.device)

    monkeypatch.setattr(mixin, "get_probability_matrix", fake_get_probability_matrix)

    bernoulli_outputs = [
        # masked_indices
        torch.tensor([[0, 1, 1, 1, 1, 0, 0]], dtype=torch.float),
        # indices_replaced: positions 1 and 2 become [MASK]
        torch.tensor([[0, 1, 1, 0, 0, 0, 0]], dtype=torch.float),
        # indices_random: among remaining masked positions, position 3 becomes random
        torch.tensor([[0, 0, 0, 1, 0, 0, 0]], dtype=torch.float),
    ]

    def fake_bernoulli(input: torch.Tensor) -> torch.Tensor:
        out = bernoulli_outputs.pop(0)
        assert out.shape == input.shape
        return out.to(device=input.device)

    monkeypatch.setattr(torch, "bernoulli", fake_bernoulli)

    random_words = torch.tensor([[0, 0, 0, 3, 0, 0, 0]])

    def fake_randint(
        high: int,
        size: tuple[int, ...],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        assert high == len(tokenizer)
        assert size == input_ids.shape
        assert dtype == torch.long
        return random_words.to(device=device)

    monkeypatch.setattr(torch, "randint", fake_randint)

    masked_input_ids, labels = mixin.mask_tokens(input_ids)

    # [CLS] [MASK] [MASK] ##world friend [SEP] [PAD]
    expected_masked_input_ids = torch.tensor([[1, 7, 7, 3, 5, 6, 0]])

    expected_labels = torch.tensor([[-100, 2, 3, 4, 5, -100, -100]])

    assert torch.equal(input_ids, original)
    assert masked_input_ids.data_ptr() != input_ids.data_ptr()
    assert labels.data_ptr() != input_ids.data_ptr()

    assert torch.equal(masked_input_ids, expected_masked_input_ids)
    assert torch.equal(labels, expected_labels)

    assert bernoulli_outputs == []


def test_TokenPLMMixin_permute_span(monkeypatch: pytest.MonkeyPatch) -> None:
    mixin = TokenPLMMixin(
        tokenizer=tokenizer,
        plm_probability=0.6,
    )

    # [CLS] hello ##world my [SEP] friend hello ##world [SEP] [PAD]
    labels = torch.tensor([[1, 2, 3, 4, 6, 5, 2, 3, 6, 0]])

    original = labels.clone()

    randperm_outputs = [
        torch.tensor([2, 0, 1]),  # Select local positions [2, 0] -> values [4, 2]
        torch.tensor([1, 0]),  # Swap selected values [2, 4] moved to [2, 0] positions
    ]

    def fake_randperm(n: int, device: torch.device | None = None) -> torch.Tensor:
        out = randperm_outputs.pop(0)
        assert out.numel() == n
        return out.to(device=device)

    monkeypatch.setattr(torch, "randperm", fake_randperm)

    mixin._permute_span(  # pyright: ignore[reportPrivateUsage]
        labels=labels,
        batch_idx=0,
        start=1,
        end=4,
    )

    # [CLS] my ##world hello [SEP] friend hello ##world [SEP] [PAD]
    expected = torch.tensor([[1, 4, 3, 2, 6, 5, 2, 3, 6, 0]])

    assert torch.equal(labels, expected)

    # Tokens outside the span must stay unchanged
    assert torch.equal(labels[0, :1], original[0, :1])
    assert torch.equal(labels[0, 4:], original[0, 4:])

    # The span must contain the same tokens, only reordered
    assert sorted(labels[0, 1:4].tolist()) == sorted(original[0, 1:4].tolist())

    # All mocked random permutations must be consumed
    assert randperm_outputs == []


def test_TokenPLMMixin_permute_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    mixim = TokenPLMMixin(
        tokenizer=tokenizer,
        plm_probability=0.6,
    )

    # [CLS] hello ##world my [SEP] friend hello ##world [SEP] [PAD]
    input_ids = torch.tensor([[1, 2, 3, 4, 6, 5, 2, 3, 6, 0]])

    original_input_ids = input_ids.clone()

    randperm_outputs = [
        torch.tensor([2, 0, 1]),  # First sent.: select local positions [2, 0] ([4, 2])
        torch.tensor([1, 0]),  # First sent.: swap selected values ([2, 4])
        torch.tensor([1, 2, 0]),  # Second sent.: select local positions [1, 2] ([2, 3])
        torch.tensor([1, 0]),  # Second sent.: swap selected values ([3, 2])
    ]

    def fake_randperm(n: int, device: torch.device | None = None) -> torch.Tensor:
        out = randperm_outputs.pop(0)
        assert out.numel() == n
        return out.to(device=device)

    monkeypatch.setattr(torch, "randperm", fake_randperm)

    result = mixim.permute_tokens(input_ids)

    # [CLS] my ##world hello [SEP] friend ##world hello [SEP] [PAD]
    expected = torch.tensor([[1, 4, 3, 2, 6, 5, 3, 2, 6, 0]])

    assert torch.equal(result, expected)

    # The original input must not be mutated
    assert torch.equal(input_ids, original_input_ids)

    # The result must be a new tensor
    assert result.data_ptr() != input_ids.data_ptr()

    # Special tokens must stay unchanged
    token_ids = cast(list[int], input_ids[0].tolist())

    special_mask = torch.tensor(
        [(token_id in tokenizer.all_special_ids) for token_id in token_ids]
    )

    assert torch.equal(result[0, special_mask], input_ids[0, special_mask])

    # Each sentence must preserve the same token set
    assert sorted(result[0, 1:4].tolist()) == sorted(input_ids[0, 1:4].tolist())
    assert sorted(result[0, 5:8].tolist()) == sorted(input_ids[0, 5:8].tolist())

    # Padding must stay unchanged
    assert result[0, 9].item() == tokenizer.pad_token_id

    # All mocked random permutations must be consumed
    assert randperm_outputs == []


def test_WordMixin_get_word_groups() -> None:
    mixin = WordMixin(tokenizer=tokenizer, subword_prefix="##")

    input_ids = torch.tensor([1, 2, 3, 4, 5, 6, 0])
    groups = mixin._get_word_groups(input_ids)  # pyright: ignore[reportPrivateUsage]

    assert groups == [
        [1, 2],  # hello ##world
        [3],  # my
        [4],  # friend
    ]


def test_WordPLMMixin_permute_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    mixin = WordPLMMixin(tokenizer=tokenizer)

    # [CLS] hello ##world well #come my friend [SEP] [PAD]
    input_ids = torch.tensor([[1, 2, 3, 8, 9, 4, 5, 6, 0]])

    original_input_ids = input_ids.clone()

    randperm_outputs = [
        torch.tensor([0, 1]),  # First groups: [0, 1] ([2, 3], [8, 9])
        torch.tensor([1, 0]),  # First groups: swap values ([8, 9], [2, 3])
        torch.tensor([0, 1]),  # Second groups: select local positions [0, 1] ([4, 5])
        torch.tensor([1, 0]),  # Second groups: swap values ([5, 4])
    ]

    def fake_randperm(n: int, device: torch.device | None = None) -> torch.Tensor:
        out = randperm_outputs.pop(0)
        assert out.numel() == n
        return out.to(device=device)

    monkeypatch.setattr(torch, "randperm", fake_randperm)

    result = mixin.permute_tokens(input_ids)

    # [CLS] well #come hello ##world friend my [SEP] [PAD]
    expected = torch.tensor([[1, 8, 9, 2, 3, 5, 4, 6, 0]])

    assert torch.equal(result, expected)

    # The original input must not be mutated
    assert torch.equal(input_ids, original_input_ids)

    # The result must be a new tensor
    assert result.data_ptr() != input_ids.data_ptr()

    # Special tokens must stay unchanged
    token_ids = cast(list[int], input_ids[0].tolist())

    special_mask = torch.tensor(
        [(token_id in tokenizer.all_special_ids) for token_id in token_ids]
    )

    assert torch.equal(result[0, special_mask], input_ids[0, special_mask])

    # Each sentence must preserve the same token set
    assert sorted(result[0, 1:7].tolist()) == sorted(input_ids[0, 1:7].tolist())

    # Padding must stay unchanged
    assert result[0, 8].item() == tokenizer.pad_token_id

    # All mocked random permutations must be consumed
    assert randperm_outputs == []


@pytest.mark.parametrize(
    "collator_cls",
    [
        DataCollatorForTokenMLM,
        DataCollatorForTokenMPLM,
        DataCollatorForTokenMLMAndWordPLM,
    ],
)
def test_TokenMLM_collate_batch_from_lists(
    collator_cls: type[DataCollatorForTokenMLM],
) -> None:
    collator = collator_cls(tokenizer=tokenizer)

    examples: list[dict[str, Any] | list[int]] = [
        [1, 2, 3, 6],
        [1, 4, 5, 6, 0],
    ]

    batch = collator._collate_batch(examples)  # pyright: ignore[reportPrivateUsage]

    expected_input_ids = torch.tensor(
        [
            [1, 2, 3, 6, 0],
            [1, 4, 5, 6, 0],
        ]
    )

    assert set(batch.keys()) == {"input_ids"}
    assert torch.equal(batch["input_ids"], expected_input_ids)


@pytest.mark.parametrize(
    "collator_cls",
    [
        DataCollatorForTokenMLM,
        DataCollatorForTokenMPLM,
        DataCollatorForTokenMLMAndWordPLM,
    ],
)
def test_TokenMLM_collate_batch_from_dicts(
    collator_cls: type[DataCollatorForTokenMLM],
) -> None:
    collator = collator_cls(tokenizer=tokenizer)

    examples: list[dict[str, Any]] = [
        {"input_ids": [1, 2, 3, 6]},
        {"input_ids": [1, 4, 5, 6, 0]},
    ]

    batch = collator._collate_batch(examples)  # pyright: ignore[reportPrivateUsage]

    expected_input_ids = torch.tensor(
        [
            [1, 2, 3, 6, 0],
            [1, 4, 5, 6, 0],
        ]
    )

    assert set(batch.keys()) == {"input_ids"}
    assert torch.equal(batch["input_ids"], expected_input_ids)


@pytest.mark.parametrize(
    "collator_cls",
    [
        DataCollatorForTokenMLM,
    ],
)
def test_TokenMLM_collate_batch_padding(
    collator_cls: type[DataCollatorForTokenMLM],
) -> None:
    collator = collator_cls(tokenizer=tokenizer)

    examples: list[dict[str, Any]] = [
        {"input_ids": [1, 2, 3, 6]},
        {"input_ids": [1, 4, 5, 6, 0]},
    ]

    batch = collator._collate_batch(examples)  # pyright: ignore[reportPrivateUsage]
    assert set(batch.keys()) == {"input_ids"}

    expected_input_ids = torch.tensor(
        [
            [1, 2, 3, 6, 0],
            [1, 4, 5, 6, 0],
        ]
    )
    assert torch.equal(batch["input_ids"], expected_input_ids)


@pytest.mark.parametrize(
    "collator_cls",
    [
        DataCollatorForTokenMLM,
        DataCollatorForTokenMPLM,
        DataCollatorForTokenMLMAndWordPLM,
        DataCollatorForWordMLM,
        DataCollatorForWordMLMAndTokenPLM,
        DataCollatorForWordMPLM,
    ],
)
def test_collate_batch_from_dicts_with_word_ids(
    collator_cls: type[DataCollatorForTokenMLM],
) -> None:
    collator = collator_cls(tokenizer=tokenizer)

    examples: list[dict[str, Any]] = [
        {"input_ids": [1, 2, 3, 6], "word_ids": [None, 0, 0, None]},
        {"input_ids": [1, 4, 5, 6, 0], "word_ids": [None, 0, 1, None, None]},
    ]

    batch = collator._collate_batch(examples)  # pyright: ignore[reportPrivateUsage]

    expected_input_ids = torch.tensor(
        [
            [1, 2, 3, 6, 0],
            [1, 4, 5, 6, 0],
        ]
    )

    expected_word_ids = torch.tensor(
        [
            [-1, 0, 0, -1, -1],
            [-1, 0, 1, -1, -1],
        ]
    )

    assert set(batch.keys()) == {"input_ids", "word_ids"}
    assert torch.equal(batch["input_ids"], expected_input_ids)
    assert torch.equal(batch["word_ids"], expected_word_ids)


@pytest.mark.parametrize(
    "collator_cls",
    [
        DataCollatorForTokenMLM,
        DataCollatorForTokenMPLM,
        DataCollatorForTokenMLMAndWordPLM,
    ],
)
def test_TokenMLM_torch_call_without_mlm(
    collator_cls: type[DataCollatorForTokenMLM],
) -> None:
    collator = collator_cls(
        tokenizer=tokenizer,
        mlm=False,
        label_pad_token_id=-100,
    )

    examples: list[dict[str, Any] | list[int]] = [
        [1, 2, 3, 6],
        [1, 4, 5, 6, 0],
    ]

    batch = collator.torch_call(examples)

    expected_input_ids = torch.tensor(
        [
            [1, 2, 3, 6, 0],
            [1, 4, 5, 6, 0],
        ]
    )

    expected_labels = torch.tensor(
        [
            [1, 2, 3, 6, -100],
            [1, 4, 5, 6, -100],
        ]
    )

    assert torch.equal(batch["input_ids"], expected_input_ids)
    # assert torch.equal(batch["original_input_ids"], expected_input_ids)
    assert torch.equal(batch["labels"], expected_labels)

    # assert batch["original_input_ids"].data_ptr() != batch["input_ids"].data_ptr()
    assert batch["labels"].data_ptr() != batch["input_ids"].data_ptr()


@pytest.mark.parametrize(
    "collator_cls",
    [
        DataCollatorForTokenMLM,
        DataCollatorForTokenMPLM,
        DataCollatorForTokenMLMAndWordPLM,
    ],
)
def test_TokenMLM_torch_call_with_mlm(
    monkeypatch: pytest.MonkeyPatch,
    collator_cls: type[DataCollatorForTokenMLM],
) -> None:
    collator = collator_cls(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=1.0,
        label_pad_token_id=-100,
    )

    examples: list[dict[str, Any] | list[int]] = [
        [1, 2, 3, 6],
        [1, 4, 5, 6, 0],
    ]

    expected_original_input_ids = torch.tensor(
        [
            [1, 2, 3, 6, 0],
            [1, 4, 5, 6, 0],
        ]
    )

    fake_masked_input_ids = torch.tensor(
        [
            [1, 7, 7, 6, 0],
            [1, 7, 5, 6, 0],
        ]
    )

    fake_labels = torch.tensor(
        [
            [-100, 2, 3, -100, -100],
            [-100, 4, 5, -100, -100],
        ]
    )

    def fake_mask_tokens(
        input_ids: torch.Tensor,
        word_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert torch.equal(input_ids, expected_original_input_ids)
        return fake_masked_input_ids.clone(), fake_labels.clone()

    monkeypatch.setattr(collator, "mask_tokens", fake_mask_tokens)

    batch = collator.torch_call(examples)

    assert torch.equal(batch["input_ids"], fake_masked_input_ids)
    # assert torch.equal(batch["original_input_ids"], expected_original_input_ids)
    assert torch.equal(batch["labels"], fake_labels)

    # assert batch["original_input_ids"].data_ptr() != batch["input_ids"].data_ptr()


@pytest.mark.parametrize(
    "collator_cls",
    [
        DataCollatorForTokenMLM,
        DataCollatorForTokenMPLM,
        DataCollatorForTokenMLMAndWordPLM,
    ],
)
def test_TokenMLM_call_uses_torch_call(
    monkeypatch: pytest.MonkeyPatch,
    collator_cls: type[DataCollatorForTokenMLM],
) -> None:
    collator = collator_cls(tokenizer=tokenizer)

    examples: list[dict[str, Any] | list[int]] = [
        [1, 2, 3, 6],
    ]

    expected_batch = {
        "input_ids": torch.tensor([[1, 2, 3, 6]]),
    }

    def fake_torch_call(
        examples_: list[dict[str, Any] | list[int]],
    ) -> dict[str, torch.Tensor]:
        assert examples_ == examples
        return expected_batch

    monkeypatch.setattr(collator, "torch_call", fake_torch_call)

    batch = collator(examples)

    assert batch is expected_batch


@pytest.mark.parametrize(
    "collator_cls",
    [
        DataCollatorForWordMLM,
        DataCollatorForWordMLMAndTokenPLM,
        DataCollatorForWordMPLM,
    ],
)
def test_WordMLM_get_probability_matrix(
    monkeypatch: pytest.MonkeyPatch,
    collator_cls: type[DataCollatorForTokenMLM],
) -> None:
    collator = collator_cls(
        tokenizer=tokenizer,
        mlm_probability=0.34,
    )

    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 0]])

    randperm_outputs = [
        torch.tensor([0, 2, 1]),
    ]

    def fake_randperm(
        n: int,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        out = randperm_outputs.pop(0)
        return out.to(device=device)

    monkeypatch.setattr(torch, "randperm", fake_randperm)

    probability_matrix = collator.get_probability_matrix(input_ids)

    expected = torch.tensor([[0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]])

    assert torch.equal(probability_matrix, expected)
    assert randperm_outputs == []


examples: list[dict[str, list[int]]] = [
    {
        "input_ids": [
            1,  # CLS
            2,  # hello
            3,  # ##world
            4,  # my
            5,  # friend
            6,  # SEP
        ]
    }
]


@pytest.mark.parametrize(
    "collator_cls",
    [
        DataCollatorForTokenMLM,
        DataCollatorForWordMLM,
        DataCollatorForTokenMPLM,
        DataCollatorForTokenMLMAndWordPLM,
        DataCollatorForWordMLMAndTokenPLM,
        DataCollatorForWordMPLM,
    ],
)
def test_MLM_and_PLM(collator_cls: type[DataCollatorForTokenMLM]) -> None:
    collator = collator_cls(tokenizer=tokenizer)

    batch = collator(examples)

    assert "input_ids" in batch
    assert "labels" in batch
    # assert "original_input_ids" in batch
    assert batch["input_ids"].shape == batch["labels"].shape


@pytest.mark.parametrize(
    "collator_cls",
    [
        DataCollatorForTokenMLM,
        DataCollatorForWordMLM,
        DataCollatorForTokenMPLM,
        DataCollatorForTokenMLMAndWordPLM,
        DataCollatorForWordMLMAndTokenPLM,
        DataCollatorForWordMPLM,
    ],
)
def test_MLM(collator_cls: type[DataCollatorForTokenMLM]) -> None:
    collator = collator_cls(
        tokenizer=tokenizer,
        max_length=6,
    )

    examples = [
        {"input_ids": [1, 2, 3, 6]},
        {"input_ids": [1, 4, 6]},
    ]

    batch = collator(examples)

    assert batch["input_ids"].shape == (2, 6)


@pytest.mark.parametrize(
    "collator_cls",
    [
        DataCollatorForTokenMPLM,
        DataCollatorForTokenMLMAndWordPLM,
        DataCollatorForWordMLMAndTokenPLM,
        DataCollatorForWordMPLM,
    ],
)
def test_PLM(collator_cls: type[DataCollatorForTokenMLM]) -> None:
    collator = collator_cls(tokenizer=tokenizer)

    batch = collator(examples)

    assert "plm_labels" in batch


def test_TokenMPLM() -> None:
    collator = DataCollatorForTokenMPLM(
        tokenizer=tokenizer,
        mlm_probability=0.15,
        plm_probability=1.0,
    )

    batch = collator(examples)

    collator = DataCollatorForTokenMPLM(
        tokenizer=tokenizer,
        plm_probability=1.0,
    )

    batch = collator(examples)

    original = torch.tensor([1, 2, 3, 4, 5, 6])

    permuted = batch["plm_labels"][0]

    assert not torch.equal(original, permuted)


def test_WordMPLM() -> None:
    collator = DataCollatorForWordMPLM(
        tokenizer=tokenizer,
        mlm_probability=0.15,
        plm_probability=0.6,
    )

    batch = collator(examples)

    collator = DataCollatorForWordMPLM(
        tokenizer=tokenizer,
        plm_probability=1.0,
    )

    batch = collator(examples)

    masked = batch["input_ids"][0]
    permuted = batch["plm_labels"][0]

    if (masked == 2).any().item() and (masked == 3).any().item():
        hello_idx = (permuted == 2).nonzero().item()
        subword_idx = (permuted == 3).nonzero().item()

        assert abs(hello_idx - subword_idx) == 1


@pytest.mark.parametrize(
    "collator_cls",
    [
        DataCollatorForTokenMLM,
        DataCollatorForTokenMPLM,
        DataCollatorForTokenMLMAndWordPLM,
    ],
)
def test_TokenMLM(collator_cls: type[DataCollatorForTokenMLM]) -> None:
    """mlm_probability = 0.0"""
    collator = collator_cls(
        tokenizer=tokenizer,
        mlm_probability=0.0,
    )

    batch = collator(examples)

    expected = torch.full_like(
        batch["labels"],
        fill_value=-100,
    )

    assert torch.equal(batch["labels"], expected)

    """ mlm_probability = 1.0 """
    collator = collator_cls(
        tokenizer=tokenizer,
        mlm_probability=1.0,
    )

    batch = collator(examples)

    labels = batch["labels"][0]

    expected = torch.tensor(
        [
            -100,  # CLS
            2,
            3,
            4,
            5,
            -100,  # SEP
        ]
    )

    assert torch.equal(labels, expected)


@pytest.mark.parametrize(
    "collator_cls",
    [
        DataCollatorForWordMLM,
        DataCollatorForWordMLMAndTokenPLM,
        DataCollatorForWordMPLM,
    ],
)
def test_WordMLM(collator_cls: type[DataCollatorForTokenMLM]) -> None:
    collator = collator_cls(
        tokenizer=tokenizer,
        mlm_probability=1.0,
    )

    batch = collator(examples)

    labels = batch["labels"][0]

    hello_word = labels[1:3]

    assert torch.all(hello_word != -100)


@pytest.mark.parametrize(
    "collator_cls",
    [
        DataCollatorForTokenMPLM,
        DataCollatorForWordMLMAndTokenPLM,
    ],
)
def test_TokenPLM(collator_cls: type[DataCollatorForTokenMLM]) -> None:
    collator = collator_cls(
        tokenizer=tokenizer,
        mlm_probability=1.0,
        plm_probability=1.0,  # pyright: ignore[reportCallIssue]
    )

    batch = collator(examples)

    collator = collator_cls(
        tokenizer=tokenizer,
        mlm_probability=1.0,
    )

    batch = collator(examples)

    labels = batch["labels"][0]

    assert (labels[1] == -100) == (labels[2] == -100)


@pytest.mark.parametrize(
    "collator_cls",
    [
        DataCollatorForTokenMLMAndWordPLM,
        DataCollatorForWordMPLM,
    ],
)
def test_WordPLM(collator_cls: type[DataCollatorForTokenMLM]) -> None:
    collator = collator_cls(
        tokenizer=tokenizer,
        plm_probability=1.0,  # pyright: ignore[reportCallIssue]
    )

    batch = collator(examples)

    masked = batch["input_ids"][0]
    permuted = batch["plm_labels"][0]

    if (masked == 2).any().item() and (masked == 3).any().item():
        hello_idx = (permuted == 2).nonzero().item()
        subword_idx = (permuted == 3).nonzero().item()

        assert abs(hello_idx - subword_idx) == 1
