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
from torcheval.metrics.functional import multiclass_confusion_matrix
import numpy as np
import math
import safetensors

def TeClassifierBatchPredict(model,dataset,batch_size):
  
  device=('cuda' if torch.cuda.is_available() else 'cpu')
  
  if device=="cpu":
    dtype=torch.float64
    model.to(device,dtype=dtype)
  else:
    dtype=torch.double
    model.to(device,dtype=dtype)
    
  model.eval()
  predictionloader=torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False)

  with torch.no_grad():
    iteration=0
    for batch in predictionloader:
      inputs=batch["input"]
      inputs = inputs.to(device,dtype=dtype)
      predictions=model(inputs,prediction_mode=True)
      
      if iteration==0:
        predictions_list=predictions.to("cpu")
      else:
        predictions_list=torch.concatenate((predictions_list,predictions.to("cpu")), axis=0, out=None)
      iteration+=1
  
  return predictions_list
      
      
def TeProtoNetClassifierBatchPredict(model,dataset,batch_size,embeddings_s,classes_s,prediction_mode=True):
  
  device=('cuda' if torch.cuda.is_available() else 'cpu')
  
  if device=="cpu":
    dtype=torch.float64
    model.to(device,dtype=dtype)
  else:
    dtype=torch.double
    model.to(device,dtype=dtype)
    
  model.eval()
  predictionloader=torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False)

  with torch.no_grad():
    iteration=0
    if not (embeddings_s==None and classes_s==None):
      embeddings_s = embeddings_s["input"][range(0,len(embeddings_s))]
      embeddings_s = embeddings_s.to(device,dtype=dtype)
      classes_s = classes_s.to(device,dtype=dtype)
    for batch in predictionloader:
      inputs=batch["input"]
      inputs = inputs.to(device,dtype=dtype)
      results=model(input_q=inputs,classes_q=None,input_s=embeddings_s,classes_s=classes_s,prediction_mode=prediction_mode)
      
      if prediction_mode==True:
        if iteration==0:
          predictions_list=results.to("cpu")
        else:
          predictions_list=torch.concatenate((predictions_list,results.to("cpu")), axis=0, out=None)
        iteration+=1
      else:
        if iteration==0:
          predictions_list=results[0].to("cpu")
          distances_list=results[1].to("cpu")
          query_classes_list="None"
          query_embeddings_list =results[3].to("cpu")
          prototypes=results[4].to("cpu")
        else:
          predictions_list=torch.concatenate((predictions_list,results[0].to("cpu")), axis=0, out=None)
          distances_list=torch.concatenate((distances_list,results[1].to("cpu")), axis=0, out=None)
          query_classes_list="None"
          query_embeddings_list=torch.concatenate((query_embeddings_list,results[3].to("cpu")), axis=0, out=None)
        iteration+=1
        
  if prediction_mode==False:
    return predictions_list,distances_list,query_classes_list,query_embeddings_list,prototypes    
  else:
    return predictions_list


