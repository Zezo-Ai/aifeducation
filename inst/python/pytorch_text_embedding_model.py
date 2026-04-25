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


import transformers
import tokenizers
import torch

class IdentityTransformer(torch.nn.Module):
  def __init__(self,num_layer):
    super().__init__()
    self.num_layer=num_layer
  
  def forward(self,input_ids,attention_mask,token_type_ids=None):
    hidden_states_of_layers=()
    for i in range(self.num_layer+1):
      hidden_states_of_layers=hidden_states_of_layers+tuple(input_ids)
    hidden_states={"hidden_states": hidden_states_of_layers}
    return hidden_states

class TextEmbeddingModel(torch.nn.Module):
  def __init__(self,base_model,chunks, emb_layer_min, emb_layer_max, emb_pool_type, pad_value,sequence_mode):
    super().__init__()
    self.base_model=base_model
    self.chunks=chunks
    self.emb_layer_min=emb_layer_min
    self.emb_layer_max=emb_layer_max
    self.emb_pool_type=emb_pool_type
    self.pad_value=pad_value
    self.sequence_mode=sequence_mode
    self.n_layers=emb_layer_max-emb_layer_min+1

  @torch.inference_mode()
  def forward(self,input_ids,attention_mask,token_type_ids=None):
    #Select relevant chunks for the case that more chunks are available (e.g. long documents)
    n_chunks=min(input_ids.size(0),self.chunks)
    index=torch.arange(start=0,end=n_chunks).to(device=input_ids.device,dtype=torch.int)
    input_ids=torch.index_select(input_ids,dim=0,index=index)
    attention_mask=torch.index_select(attention_mask,dim=0,index=index)
    if not token_type_ids is None:
      token_type_ids=torch.index_select(token_type_ids,dim=0,index=index)
    
    # Apply the model and receive the hidden states for calculating embeddings    
    if token_type_ids is None:
      embeddings=self.base_model(input_ids=input_ids,attention_mask=attention_mask,output_hidden_states =True)
    else:
      embeddings=self.base_model(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids,output_hidden_states =True)
    
    # differentiate between hidden states which all have the same sequence length
    if self.sequence_mode=="equal":
      #Create a tensor of the hidden states for fast proccessing
      
      relevant_embeddings=torch.stack(embeddings["hidden_states"][self.emb_layer_min:(self.emb_layer_max+1)],dim=1)
      #Continue depending on the chosen pooling method
      if self.emb_pool_type=="Average":
        masked_expanded=torch.unsqueeze(attention_mask,dim=2)
        masked_expanded=masked_expanded.expand(embeddings["hidden_states"][0].size())
        masked_expanded=torch.unsqueeze(masked_expanded,dim=1)
        masked_expanded=masked_expanded.expand(relevant_embeddings.size())
        
        sum_over_sequences=torch.sum(masked_expanded*relevant_embeddings,dim=2)
        sum_over_layers=torch.sum(sum_over_sequences,dim=1)
        
        n_elements=torch.unsqueeze(self.n_layers*torch.sum(attention_mask,dim=1),dim=1).expand(sum_over_layers.size())
        final_embeddings=sum_over_layers/n_elements
      elif self.emb_pool_type=="CLS":
        cls_tokens=relevant_embeddings.select(dim=2,index=0)
        final_embeddings=torch.sum(cls_tokens,dim=(1))/self.n_layers
    else:
      if self.emb_pool_type=="CLS":
        cls_tokens=torch.zeros(embeddings["hidden_states"][0].size(0),embeddings["hidden_states"][0].size(2)).to(embeddings["hidden_states"][0].device)
        for i in range(self.emb_layer_min,self.emb_layer_max+1):
          tmp_states=torch.squeeze(torch.index_select(embeddings["hidden_states"][i],dim=1,index=torch.zeros(1).to(embeddings["hidden_states"][0].device,torch.int)),dim=1)
          cls_tokens=cls_tokens+tmp_states
        cls_tokens=cls_tokens/self.n_layers
        final_embeddings=cls_tokens

    #Add missing rows to ensure embeddings of the same shape
    if n_chunks<self.chunks:
      additional_row=torch.ones((self.chunks-n_chunks,final_embeddings.size(1)))*self.pad_value
      additional_row=additional_row.to(final_embeddings.device,final_embeddings.dtype)
      final_embeddings=torch.cat((final_embeddings,additional_row),dim=0)
    
    final_embeddings=torch.unsqueeze(final_embeddings,dim=0)
    return final_embeddings
    
