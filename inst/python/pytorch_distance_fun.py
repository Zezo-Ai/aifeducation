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

import torch 
import numpy as np

# CosineDistance for all possible pairs
def CosineDistance(
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 1e-8,
):
    """
    Memory-efficient cosine distance.

    x: (T, F) or (B, T, F)
    y: (V, F) or (B, V, F)
    returns: (T, V) or (B, T, V)
    """
    x = torch.nn.functional.normalize(x, p=2, dim=-1, eps=eps)  # L2 normalize: x / ||x||
    y = torch.nn.functional.normalize(y, p=2, dim=-1, eps=eps)  # L2 normalize: y / ||y||

    # cosine distance = 1 - cosine similarity
    # since normalized: cosine(x, y) = x @ y^T
    return 1.0 - torch.matmul(x, y.transpose(-1, -2))

