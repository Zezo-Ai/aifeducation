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


from transformers import DataCollatorForLanguageModeling

from math import ceil
# ======================================================================================
# DataCollator
# ======================================================================================

class CollatorMaker_PT:
    def __init__(self, tokenizer,max_length, transformers_version_controll=0, mlm = True, mlm_probability = 0.15, plm_probability = 0.6, mask_whole_words = False):
        self.mask_whole_words = mask_whole_words
        self.max_length = max_length
        self.transformers_version_controll = transformers_version_controll
        self.collator = self.__make_collator()(tokenizer = tokenizer,
                                               transformers_version_controll=transformers_version_controll,
                                               max_length=self.max_length,
                                               mlm = mlm,
                                               mlm_probability = mlm_probability, 
                                               plm_probability = plm_probability)
        
    def __get_class_type(self):
        if self.transformers_version_controll<2:
          return DataCollatorForWholeWordMask if self.mask_whole_words \
                 else DataCollatorForLanguageModeling
        else:
          return DataCollatorForLanguageModeling
    
    def __make_collator(self):
        base_class = self.__get_class_type()

        class DataCollatorForMPLM(base_class):
            def __init__(self, tokenizer,transformers_version_controll, mlm, mlm_probability, plm_probability,max_length,mask_whole_words=self.mask_whole_words):
                self.transformers_version_controll=transformers_version_controll
                self.max_length=max_length
                self.mask_whole_words=mask_whole_words
                if isinstance(base_class, DataCollatorForLanguageModeling):
                  if self.transformers_version_controll<=0:
                    super().__init__(tokenizer = tokenizer, mlm = mlm, mlm_probability = mlm_probability)
                  elif self.transformers_version_controll==1:
                    super().__init__(tokenizer = tokenizer, mlm = mlm, mlm_probability = mlm_probability,mask_replace_prob = 1.0,random_replace_prob = 0.0)
                  else:
                    super().__init__(tokenizer = tokenizer, mlm = mlm, whole_word_mask = self.mask_whole_words, mlm_probability = mlm_probability, mask_replace_prob = 1.0,random_replace_prob = 0.0)
                else:
                  if self.transformers_version_controll==0:
                    super().__init__(tokenizer = tokenizer,  mlm_probability = mlm_probability)
                  elif self.transformers_version_controll==1:
                    super().__init__(tokenizer = tokenizer,  mlm_probability = mlm_probability, mask_replace_prob = 1.0,random_replace_prob = 0.0)
                self.tokenizer = tokenizer
                self.plm_probability = plm_probability
                if self.transformers_version_controll==2:
                  self.seed=10
                  self.generator = self.get_generator(self.seed)

            def make_plm_labels(self, input_ids):
                plm_labels = input_ids.clone()

                # Per each i-sample
                for i in range(input_ids.size(0)):
                    # Indices of SEP tokens in i-sample
                    sep_pos_ids = (input_ids[i] == self.tokenizer.sep_token_id).nonzero().view(-1)

                    # Per each j-sentence in i-sample
                    for j in range(sep_pos_ids.size(0)):
                        sent_start_id = 0 if j == 0 else sep_pos_ids[j - 1].item()
                        sent_end_id = sep_pos_ids[j].item()

                        # Get indices of all elements in j-th sentence (exclude sent_start_id and sent_end_id)
                        sent_ids = torch.arange(sent_start_id + 1, sent_end_id)
                        sent_size = sent_ids.size(0)

                        # Number of tokens to permute
                        k = ceil(sent_size * self.plm_probability)

                        if k <= 1:
                            continue
                        
                        # Randomly select K indices from all available indices
                        selected_indices = sent_ids[torch.randperm(sent_size)[:k]]

                        # Get the values to permute
                        values_to_permute = plm_labels[i, selected_indices].clone()

                        # Permute those values
                        permuted_values = values_to_permute[torch.randperm(len(values_to_permute))]

                        # Replace the values in the original tensor with the permuted values
                        plm_labels[i, selected_indices] = permuted_values
                return plm_labels
            
            def collate_batch(self, examples):
                """
                examples: List(Dict)
                Every dictionary has the following keys:
                - input_ids
                - attention_mask

                examples
                [
                    {
                        'input_ids': tensor([]),
                        'attention_mask': tensor([])
                    },
                    {
                        'input_ids': tensor([]),
                        'attention_mask': tensor([])
                    }, ...
                ]
                """

                # self(examples) executes DataCollatorForLanguageModeling/DataCollatorForWholeWordMask
                # added masking <mask>/30526
                input_ids = self(examples)["input_ids"]
                # attention_mask = self(examples)["attention_mask"]
                
                """
                MPNet: Tokens with indices e.g. 30526 set to -100 are ignored (masked)
                """
                mlm_labels = input_ids.clone()
                mlm_labels[input_ids == self.tokenizer.mask_token_id] = -100

                # add permutation
                plm_labels = self.make_plm_labels(input_ids)

                return {
                    "input_ids": torch.tensor([i['input_ids'].tolist() for i in examples]),
                    "attention_mask": torch.tensor([i['attention_mask'].tolist() for i in examples]),
                    "mlm_labels": mlm_labels,
                    "plm_labels": plm_labels
                }
        return DataCollatorForMPLM
