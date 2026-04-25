import torch
from typing import List, Dict, Any, Union
from transformers import PreTrainedTokenizerFast
import random


class AifeDataCollatorForWholeWordMask:
  def __init__(self, 
               tokenizer: PreTrainedTokenizerFast,
               mlm_probability: float = 0.15,
               pad_input=True):
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
      raise ValueError("Tokenizer must be PreTrainedTokenizerFast")
    
    self.tokenizer = tokenizer
    self.mlm_probability = mlm_probability
    self.pad_input=pad_input

  def __call__(self, examples: List[Union[List[int], Any, dict[str, Any]]]) -> Dict[str, torch.Tensor]:
    # Gather word_ids for each example
    word_ids_list = []
    for e in examples:
      if hasattr(e, "word_ids"):  # e.g. BatchEncoding
        word_ids_list.append(e.word_ids())
      elif "word_ids" in e: 
        word_ids_list.append(e["word_ids"])
        e.pop("word_ids")
      else:
        raise ValueError("There is no information about word_ids!")

    #Remove word ids which are not part of the training
    if "word_ids" in examples:
      examples.pop("word_ids")
    
    # Gathering in one batch and padding by max sequence length
    batch = self.tokenizer.pad(
      examples,
      return_tensors = "pt",
      padding=self.pad_input
    )
    
    input_ids = batch["input_ids"]
    labels = input_ids.clone()



    # Apply whole word masking
    input_ids, labels = self.mask_whole_words(input_ids, labels, word_ids_list)

    batch["input_ids"] = input_ids
    batch["labels"] = labels

    return batch

  def mask_whole_words(self, input_ids, labels, word_ids_list):
    self.tokenizer: PreTrainedTokenizerFast
    
    batch_size, seq_len = input_ids.shape

    # Masking each line
    for i in range(batch_size):
      word_ids = word_ids_list[i]
      if word_ids is None:
        continue

      # Group tokens by words
      word_to_tokens = {}
      for j, word_id in enumerate(word_ids):
        if word_id is None:
          continue
        word_to_tokens.setdefault(word_id, []).append(j)

      # Choose the words to mask (exact number of words)
      num_to_mask = max(1, int(len(word_to_tokens) * self.mlm_probability))
      words_to_mask = random.sample(list(word_to_tokens.keys()), num_to_mask)

      for w in words_to_mask:
        for idx in word_to_tokens[w]:
          input_ids[i, idx] = self.tokenizer.mask_token_id

      mask = torch.ones(seq_len, dtype = torch.bool)
      for w in words_to_mask:
        for idx in word_to_tokens[w]:
          mask[idx] = False
      labels[i][mask] = -100

    return input_ids, labels
