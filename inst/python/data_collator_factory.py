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

from typing import Any, Literal

from data_collators import (
    DataCollatorForTokenMLM,
    DataCollatorForTokenMLMAndWordPLM,
    DataCollatorForTokenMPLM,
    DataCollatorForWordMLM,
    DataCollatorForWordMLMAndTokenPLM,
    DataCollatorForWordMPLM,
)

# _THIS_DIR = Path(__file__).resolve().parent

# if str(_THIS_DIR) not in sys.path:
#     sys.path.insert(0, str(_THIS_DIR))

_collator_type_name = Literal[
    "TokenMLM",
    "WordMLM",
    "TokenMPLM",
    "TokenMLMAndWordPLM",
    "WordMLMAndTokenPLM",
    "WordMPLM",
]

_collator_type = (
    type[DataCollatorForTokenMLM]
    | type[DataCollatorForTokenMPLM]
    | type[DataCollatorForWordMLM]
    | type[DataCollatorForTokenMLMAndWordPLM]
    | type[DataCollatorForWordMLMAndTokenPLM]
    | type[DataCollatorForWordMPLM]
)

_collator_instance = (
    DataCollatorForTokenMLM
    | DataCollatorForTokenMPLM
    | DataCollatorForWordMLM
    | DataCollatorForTokenMLMAndWordPLM
    | DataCollatorForWordMLMAndTokenPLM
    | DataCollatorForWordMPLM
)

_COLLATOR_REGISTRY: dict[_collator_type_name, _collator_type] = {
    "TokenMLM": DataCollatorForTokenMLM,  # Token-level MLM
    "WordMLM": DataCollatorForWordMLM,  # Word-level MLM
    "TokenMPLM": DataCollatorForTokenMPLM,  # Token-level MLM + token-level permutation
    "TokenMLMAndWordPLM": DataCollatorForTokenMLMAndWordPLM,  # Token-level MLM + word-level permutation
    "WordMLMAndTokenPLM": DataCollatorForWordMLMAndTokenPLM,  # Word-level MLM + token-level permutation
    "WordMPLM": DataCollatorForWordMPLM,  # Word-level MLM + word-level permutation
}


def make_collator(
    name: _collator_type_name,
    **kwargs: Any,
) -> _collator_instance:
    try:
        return _COLLATOR_REGISTRY[name](**kwargs)
    except KeyError:
        raise ValueError(f"Unknown collator: {name!r}")
