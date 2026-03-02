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

# ======================================================================================
# Fill mask pipeline for MPLM
# ======================================================================================

import transformers
from transformers import FillMaskPipeline
from packaging import version
import copy

class FillMaskPipelineForMPLM():
    def __init__(self, 
                 model, 
                 tokenizer, 
                 framework: str = "pt",
                 num_workers: int = 1,
                 binary_output: bool = False,
                 top_k: int = 5,
                 tokenizer_kwargs: dict = {"return_token_type_ids": False}):
        self.model = model
        self.tokenizer = tokenizer
        self.framework = framework
        self.num_workers = num_workers
        self.binary_output = binary_output
        self.top_k = top_k
        self.tokenizer_kwargs = tokenizer_kwargs
    
    def _make_pipeline(self, n_solutions: int = 1) -> FillMaskPipeline:
        if version.Version(transformers.__version__) < version.Version("5.0.0"):
          return FillMaskPipeline(model = self.model,
                                  tokenizer = self.tokenizer,
                                  framework = self.framework,
                                  num_workers = self.num_workers,
                                  binary_output = self.binary_output,
                                  top_k = n_solutions,
                                  tokenizer_kwargs = self.tokenizer_kwargs)
        else:
          return FillMaskPipeline(model = self.model,
                        tokenizer = self.tokenizer,
                        num_workers = self.num_workers,
                        binary_output = self.binary_output,
                        top_k = n_solutions,
                        tokenizer_kwargs = self.tokenizer_kwargs)
    
    def _prepare(self, t: str) -> str:
        """
        Modify given `t` text to have only one mask token.
        The first mask token is kept, the rest are removed.

        Return:
        A text with only one mask token.
        """

        mask_token = self.tokenizer.mask_token

        first_mask_i = t.find(mask_token)
        
        maybe_next_mask_i = t[first_mask_i + 1:].find(mask_token)
        if maybe_next_mask_i != -1:  # there is one more mask token
            next_mask_i = maybe_next_mask_i + first_mask_i + 1

            left_text = t[:next_mask_i]                          # text with one mask
                                                                 # (only with the first mask)

            right_text = t[next_mask_i:].replace(mask_token, '') # text without masks
                                                                 # (the rest of masks were removed)
        else:                       # there is only one mask token
            left_text = t[:first_mask_i]                         # text without mask
            right_text = t[first_mask_i:]                        # text with one mask
        
        return left_text + right_text
    
    def _split_text(self, t: str, amount: int) -> list[dict]:
        """
        Split `t` text into `amount` dictionaries.
        FillMaskPipeline makes `amount` predictions for the first mask token.

        Return:
        A list of dictionaries (modified output of FillMaskPipeline).
        """

        n_pipeline = self._make_pipeline(n_solutions = amount)

        text_with_one_mask = self._prepare(t)
        solutions = n_pipeline(inputs = text_with_one_mask)

        for i in range(len(solutions)):
            new_sequence = t.replace(self.tokenizer.mask_token, solutions[i]['token_str'], 1)
            solutions[i]['sequence'] = new_sequence

        return solutions

    def __call__(self, text):
        mask_token_amount = text.count(self.tokenizer.mask_token)

        # result: list(dict)
        result = self._split_text(text, self.top_k)

        rest_mask_token_amount = mask_token_amount - 1
        if rest_mask_token_amount == 0:
            return result

        result = [result]
        # result: list( list(dict) )

        # Pipeline that returns the first (the best) prediction 
        one_pipeline = self._make_pipeline()
        for i in range(1, rest_mask_token_amount + 1):
            result.append(copy.deepcopy(result[i - 1]))
            for j in range(len(result[i])):
                text_with_one_mask = self._prepare(result[i][j]['sequence'])
                solution = one_pipeline(text_with_one_mask)

                result[i][j]['score'] = solution[0]['score']
                result[i][j]['token'] = solution[0]['token']
                result[i][j]['token_str'] = solution[0]['token_str']
                result[i][j]['sequence'] = result[i][j]['sequence'].replace(self.tokenizer.mask_token, solution[0]['token_str'], 1)
        return result
