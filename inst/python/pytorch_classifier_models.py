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
import math
import safetensors

class TEClassifierSequential(torch.nn.Module):
  def __init__(self,times, features, cls_pooling_features, pad_value,n_target_levels,inc_cls_head=True,skip_connection_type="ResidualGate",cls_type="Regular",cls_pooling_type="MinMax", 
              feat_act_fct="ELU",feat_size=50,feat_bias=True,feat_dropout=0.0,feat_parametrizations="None",feat_normalization_type="LayerNorm",
              ng_conv_act_fct="ELU",ng_conv_n_layers=0,ng_conv_ks_min=2, ng_conv_ks_max=4,ng_conv_dropout=0.1, ng_conv_bias=False, ng_conv_parametrizations="None", ng_conv_residual_type="ResidualGate",ng_conv_normalization_type="LayerNorm",
              dense_act_fct="ELU",dense_n_layers=0,dense_dropout=0.0,dense_bias=False,dense_parametrizations="None", dense_residual_type="ResidualGate",dense_normalization_type="LayerNorm",
              rec_act_fct="Tanh",rec_n_layers=0,rec_type="GRU",rec_bidirectional=False,rec_dropout=0.0,rec_bias=False,rec_parametrizations="None",rec_residual_type="ResidualGate",rec_normalization_type="LayerNorm", 
              tf_act_fct="ELU",tf_dense_dim=50,tf_n_layers=0,tf_dropout_rate_1=0.0,tf_dropout_rate_2=0.0,tf_attention_type="MultiHead",tf_positional_type ="absolute",tf_num_heads=1,tf_bias=False,tf_parametrizations="None",tf_residual_type="ResidualGate",tf_normalization_type="LayerNorm", tf_normalization_position="pre",
              device=None, dtype=None):
      super().__init__()
      self.inc_cls_head=inc_cls_head
 
      self.masking_layer=masking_layer(
        pad_value=pad_value
      )
      
      if features==feat_size:
        self.features_resize_layer=layer_dropout_with_mask(p=feat_dropout, pad_value=pad_value)
      else:
        self.features_resize_layer=dense_layer_with_mask(
          input_size=features,
          pad_value=pad_value,
          times=times,
          output_size=feat_size,
          act_fct=feat_act_fct,
          dropout=feat_dropout,
          bias=feat_bias,
          parametrizations=feat_parametrizations,
          device=device, 
          dtype=dtype,
          residual_type="None",
          normalization_type=feat_normalization_type
      )
      
      
      if tf_n_layers >0:
        self.stack_tf_encoder_layer=stack_tf_encoder_layer(
          dense_dim=tf_dense_dim,
          n_layers=tf_n_layers,
          times=times, 
          features=feat_size,
          pad_value=pad_value,
          dropout_rate_1=tf_dropout_rate_1,
          dropout_rate_2=tf_dropout_rate_2,
          act_fct=tf_act_fct,
          attention_type=tf_attention_type,
          positional_embedding=tf_positional_type ,
          num_heads=tf_num_heads,
          bias=tf_bias,
          parametrizations=tf_parametrizations,
          normalization_type=tf_normalization_type,
          normalization_position=tf_normalization_position,
          device=device, 
          dtype=dtype,
          residual_type=tf_residual_type
        )
      else:
        self.stack_tf_encoder_layer=identity_layer(pad_value=pad_value,apply_masking=True)
      
      if rec_n_layers>0:
        self.stack_recurrent_layers=stack_recurrent_layers(
          times=times,
          hidden_size=feat_size,
          n_layers=rec_n_layers,
          pad_value=pad_value,
          rec_type=rec_type,
          rec_bidirectional=rec_bidirectional,
          dropout=rec_dropout,
          bias=rec_bias,
          return_sequence=True,
          parametrizations=rec_parametrizations,
          device=device, 
          dtype=dtype,
          residual_type=rec_residual_type,
          normalization_type=rec_normalization_type
        )
      else:
        self.stack_recurrent_layers=identity_layer(pad_value=pad_value,apply_masking=True)
          
      if ng_conv_n_layers>0:
        self.stack_n_gram_convolution=stack_n_gram_convolution(
          ks_min=ng_conv_ks_min,
          ks_max=ng_conv_ks_max,
          times=times,
          act_fct=ng_conv_act_fct,
          features=feat_size,
          pad_value=pad_value,
          n_layers=ng_conv_n_layers,
          bias=ng_conv_bias,
          parametrizations=ng_conv_parametrizations,
          device=device,
          dtype=dtype,
          residual_type=ng_conv_residual_type,
          normalization_type=ng_conv_normalization_type
        )
      else:
        self.stack_n_gram_convolution=identity_layer(pad_value=pad_value,apply_masking=True)
        
      if dense_n_layers>0:
        self.stack_dense_layer=stack_dense_layer(
            times=times,
            hidden_size=feat_size,
            n_layers=dense_n_layers,
            pad_value=pad_value,
            dropout=dense_dropout,
            act_fct=dense_act_fct,
            bias=dense_bias,
            normalization_type=dense_normalization_type,
            parametrizations=dense_parametrizations,
            device=device, 
            dtype=dtype,
            residual_type=dense_residual_type
            )
      else:
        self.stack_dense_layer=identity_layer(pad_value=pad_value,apply_masking=True)
        
      self.summarize_layer_time=exreme_pooling_over_time(
        times=times,
        features=feat_size,
        pooling_type=cls_pooling_type,
        pad_value=pad_value)
        
      self.summarize_layer_features=layer_adaptive_extreme_pooling_1d(
        output_size=cls_pooling_features,
        pooling_type=cls_pooling_type
      )
      
      self.residual_connection=layer_residual_connection(skip_connection_type,pad_value)  
      
      if inc_cls_head==True:
        if cls_type=="Regular":
          self.classification_head=torch.nn.Linear(
                in_features=cls_pooling_features,
                out_features=n_target_levels)
        elif cls_type=="PairwiseOrthogonal":
          self.classification_head=pairwise_orthogonal_dense(
                input_size=cls_pooling_features,
                output_size=n_target_levels,
                bias=False,
                pre_dense=False,
                device=device, 
                dtype=dtype)
        elif cls_type=="PairwiseOrthogonalDense":
          self.classification_head=pairwise_orthogonal_dense(
                input_size=cls_pooling_features,
                output_size=n_target_levels,
                bias=False,
                pre_dense=True,
                device=device, 
                dtype=dtype)

  def forward(self,x,prediction_mode=True):
    y_original,mask_original=self.masking_layer(x)
    y_original,mask_original=self.features_resize_layer(y_original,mask_original)
    y,mask=self.stack_tf_encoder_layer(y_original,mask_original)
    y,mask=self.stack_recurrent_layers(y,mask)
    y,mask=self.stack_n_gram_convolution(y,mask)
    y,mask=self.stack_dense_layer(y,mask)
    y,mask=self.residual_connection(y_original,y,mask)
    y=self.summarize_layer_time(y,get_FeatureMask_from_mask(mask,y.size(2)))
    y=self.summarize_layer_features(y)
    if self.inc_cls_head==True:
      y=self.classification_head(y)
    if prediction_mode==False:
      return y
    else:
      return torch.nn.Softmax(dim=1)(y)


class TEClassifierParallel(torch.nn.Module):
  def __init__(self,times, features, pad_value,n_target_levels,inc_cls_head=True,cls_type="regular", 
              shared_feat_layer=True,feat_act_fct="ELU",feat_size=50,feat_bias=True,feat_dropout=0.0,feat_parametrizations="None",feat_normalization_type="LayerNorm",
              ng_conv_act_fct="ELU",ng_conv_n_layers=0,ng_conv_ks_min=2, ng_conv_ks_max=4,ng_conv_dropout=0.1, ng_conv_bias=False, ng_conv_parametrizations="None", ng_conv_residual_type="ResidualGate",ng_conv_normalization_type="LayerNorm",
              dense_act_fct="ELU",dense_n_layers=0,dense_dropout=0.0,dense_bias=False,dense_parametrizations="None", dense_residual_type="ResidualGate",dense_normalization_type="LayerNorm",
              rec_act_fct="Tanh",rec_n_layers=0,rec_type="GRU",rec_bidirectional=False,rec_dropout=0.0,rec_bias=False,rec_parametrizations="None", rec_residual_type="ResidualGate",rec_normalization_type="LayerNorm",
              tf_act_fct="ELU",tf_dense_dim=50,tf_n_layers=0,tf_dropout_rate_1=0.0,tf_dropout_rate_2=0.0,tf_attention_type="MultiHead",tf_positional_type ="absolute",tf_num_heads=1,tf_bias=False,tf_parametrizations="None",tf_residual_type="ResidualGate",tf_normalization_type="LayerNorm", tf_normalization_position="pre",
              merge_attention_type="MultiHead",merge_num_heads=1,merge_normalization_type="LayerNorm",merge_pooling_type="MinMax",merge_pooling_features=2,
              device=None, dtype=None):
      super().__init__()
      self.inc_cls_head=inc_cls_head
      self.shared_feat_layer=shared_feat_layer
      self.n_streams=1
 
      self.masking_layer=masking_layer(
        pad_value=pad_value
      )
      
      if features==feat_size:
        self.features_resize_layer=layer_dropout_with_mask(p=feat_dropout, pad_value=pad_value)
      else:
        self.features_resize_layer=dense_layer_with_mask(
          input_size=features,
          pad_value=pad_value,
          output_size=feat_size,
          times=times,
          act_fct=feat_act_fct,
          dropout=feat_dropout,
          bias=feat_bias,
          parametrizations=feat_parametrizations,
          device=device, 
          dtype=dtype,
          residual_type="None",
          normalization_type=feat_normalization_type
      )

      if tf_n_layers >0:
        self.n_streams+=1
        self.stack_tf_encoder_layer=stack_tf_encoder_layer(
          dense_dim=tf_dense_dim,
          n_layers=tf_n_layers,
          times=times, 
          features=feat_size,
          dropout_rate_1=tf_dropout_rate_1,
          dropout_rate_2=tf_dropout_rate_2,
          act_fct=tf_act_fct,
          attention_type=tf_attention_type,
          positional_embedding=tf_positional_type ,
          num_heads=tf_num_heads,
          bias=tf_bias,
          parametrizations=tf_parametrizations,
          normalization_type=tf_normalization_type,
          normalization_position=tf_normalization_position,
          pad_value=pad_value,
          device=device, 
          dtype=dtype,
          residual_type=tf_residual_type
        )
        if self.shared_feat_layer==False:
          if features==feat_size:
            self.features_resize_layer_tf=layer_dropout_with_mask(p=feat_dropout, pad_value=pad_value)
          else:
            self.features_resize_layer_tf=dense_layer_with_mask(
              input_size=features,
              output_size=feat_size,
              times=times,
              act_fct=tf_act_fct,
              dropout=feat_dropout,
              bias=feat_bias,
              parametrizations=feat_parametrizations,
              pad_value=pad_value,
              device=device, 
              dtype=dtype,
              residual_type="None",
              normalization_type=feat_normalization_type
          )        
        else:
          self.features_resize_layer_tf=None
      else:
        self.stack_tf_encoder_layer=None
        self.features_resize_layer_tf=None
      
      if rec_n_layers>0:
        self.n_streams+=1
        self.stack_recurrent_layers=stack_recurrent_layers(
          times=times,
          hidden_size=feat_size,
          n_layers=rec_n_layers,
          rec_type=rec_type,
          rec_bidirectional=rec_bidirectional,
          dropout=rec_dropout,
          bias=rec_bias,
          pad_value=pad_value,
          return_sequence=True,
          parametrizations=rec_parametrizations,
          device=device, 
          dtype=dtype,
          residual_type=rec_residual_type,
          normalization_type=rec_normalization_type
        )
        if self.shared_feat_layer==False:
          if features==feat_size:
            self.features_resize_layer_rec=layer_dropout_with_mask(p=feat_dropout, pad_value=pad_value)
          else:
            self.features_resize_layer_rec=dense_layer_with_mask(
              input_size=features,
              output_size=feat_size,
              times=times,
              act_fct=rec_act_fct,
              dropout=feat_dropout,
              bias=feat_bias,
              parametrizations=feat_parametrizations,
              pad_value=pad_value,
              device=device, 
              dtype=dtype,
              residual_type="None",
              normalization_type=feat_normalization_type
              )   
        else:
          self.features_resize_layer_rec=None
      else:
        self.stack_recurrent_layers=None
        self.features_resize_layer_rec=None
          
      if ng_conv_n_layers>0:
        self.n_streams+=1
        self.stack_n_gram_convolution=stack_n_gram_convolution(
          ks_min=ng_conv_ks_min,
          ks_max=ng_conv_ks_max,
          times=times,
          act_fct=ng_conv_act_fct,
          features=feat_size,
          n_layers=ng_conv_n_layers,
          bias=ng_conv_bias,
          parametrizations=ng_conv_parametrizations,
          device=device,
          dtype=dtype,
          pad_value=pad_value,
          dropout=ng_conv_dropout,
          residual_type=ng_conv_residual_type,
          normalization_type=ng_conv_normalization_type
        )

        if self.shared_feat_layer==False:        
          if features==feat_size:
            self.features_resize_layer_conv=layer_dropout_with_mask(p=feat_dropout, pad_value=pad_value)
          else:
            self.features_resize_layer_conv=dense_layer_with_mask(
              input_size=features,
              output_size=feat_size,
              times=times,
              act_fct=ng_conv_act_fct,
              dropout=feat_dropout,
              bias=feat_bias,
              parametrizations=feat_parametrizations,
              device=device, 
              dtype=dtype,
              pad_value=pad_value,
              residual_type="None",
              normalization_type=feat_normalization_type
          )        
        else:
          self.features_resize_layer_conv=None
      else:
        self.stack_n_gram_convolution=None
        self.features_resize_layer_conv=None
        
      if dense_n_layers>0:
        self.n_streams+=1
        self.stack_dense_layer=stack_dense_layer(
            times=times,
            hidden_size=feat_size,
            n_layers=dense_n_layers,
            dropout=dense_dropout,
            act_fct=dense_act_fct,
            bias=dense_bias,
            normalization_type=dense_normalization_type,
            parametrizations=dense_parametrizations,
            device=device, 
            pad_value=pad_value,
            dtype=dtype,
            residual_type=dense_residual_type
            )

        if self.shared_feat_layer==False:            
          if features==feat_size:
            self.features_resize_layer_dense=layer_dropout_with_mask(p=feat_dropout, pad_value=pad_value)
          else:
            self.features_resize_layer_dense=dense_layer_with_mask(
              input_size=features,
              output_size=feat_size,
              times=times,
              act_fct=dense_act_fct,
              dropout=feat_dropout,
              bias=feat_bias,
              parametrizations=feat_parametrizations,
              pad_value=pad_value,
              device=device, 
              dtype=dtype,
              residual_type="None",
              normalization_type=feat_normalization_type
          )
        else:
          self.features_resize_layer_dense=None
      else:
        self.stack_dense_layer=None
        self.features_resize_layer_dense=None

        
      self.merge_layer=merge_layer(
        times=times,
        features=feat_size,
        n_extracted_features=merge_pooling_features,
        n_input_streams=self.n_streams,
        pooling_type=merge_pooling_type,
        normalization_type=merge_normalization_type,
        pad_value=pad_value,
        attention_type=merge_attention_type,
        num_heads=merge_num_heads,
        device=device,
        dtype=dtype
      )
      
      if inc_cls_head==True:
        if cls_type=="Regular":
          self.classification_head=torch.nn.Linear(
                in_features=merge_pooling_features,
                out_features=n_target_levels)
        elif cls_type=="PairwiseOrthogonal":
          self.classification_head=pairwise_orthogonal_dense(
                input_size=merge_pooling_features,
                output_size=n_target_levels,
                bias=False,
                pre_dense=False,
                device=device, 
                dtype=dtype)
        elif cls_type=="PairwiseOrthogonalDense":
          self.classification_head=pairwise_orthogonal_dense(
                input_size=merge_pooling_features,
                output_size=n_target_levels,
                bias=False,
                pre_dense=True,
                device=device, 
                dtype=dtype)

  def forward(self,x,prediction_mode=True):
    y,mask=self.masking_layer(x)
    
    y_original,mask_original=self.features_resize_layer(y,mask)
    tensor_list=[y_original.clone()]
    
    if not self.stack_tf_encoder_layer==None:
      if self.shared_feat_layer==False:
        tmp_tf,mask_tf=self.features_resize_layer_tf(y,mask)
      else:
        tmp_tf,mask_tf=self.features_resize_layer(y,mask)
      tmp_tf,mask_tf=self.stack_tf_encoder_layer(tmp_tf,mask_tf)
      tensor_list.append(tmp_tf.clone())
      
    if not self.stack_recurrent_layers==None:
      if self.shared_feat_layer==False:
        tmp_rec,mask_rec=self.features_resize_layer_rec(y,mask)
      else:
        tmp_rec,mask_rec=self.features_resize_layer(y,mask)
      tmp_rec,mask_rec=self.stack_recurrent_layers(tmp_rec,mask_rec)
      tensor_list.append(tmp_rec.clone())
      
    if not self.stack_n_gram_convolution==None:
      if self.shared_feat_layer==False:
        tmp_ng,mask_ng=self.features_resize_layer_conv(y,mask)
      else:
        tmp_ng,mask_ng=self.features_resize_layer(y,mask)
      tmp_ng,mask_ng=self.stack_n_gram_convolution(tmp_ng,mask_ng)
      tensor_list.append(tmp_ng.clone())
      
    if not self.stack_dense_layer==None:
      if self.shared_feat_layer==False:
        tmp_dense,mask_dense=self.features_resize_layer_dense(y,mask)
      else:
        tmp_dense,mask_dense=self.features_resize_layer(y,mask)
      tmp_dense,mask_dense=self.stack_dense_layer(tmp_dense,mask_dense)
      tensor_list.append(tmp_dense.clone())
    
    #Merge  
    result=self.merge_layer(tensor_list,mask_original)
    if self.inc_cls_head==True:
      result=self.classification_head(result)
    if prediction_mode==False:
      return result
    else:
      return torch.nn.Softmax(dim=1)(result)


class TEClassifierPrototype(torch.nn.Module):
  def __init__(self,times, features, pad_value,target_levels,core_net_type,projection_type="Regular",embedding_dim=2,skip_connection_type="ResidualGate",inc_cls_head=True,cls_type="regular", 
              shared_feat_layer=True, feat_act_fct="ELU",feat_size=50,feat_bias=True,feat_dropout=0.0,feat_parametrizations="None",feat_normalization_type="LayerNorm",
              ng_conv_act_fct="ELU",ng_conv_n_layers=0,ng_conv_ks_min=2, ng_conv_ks_max=4,ng_conv_dropout=0.1, ng_conv_bias=False, ng_conv_parametrizations="None", ng_conv_residual_type="ResidualGate",ng_conv_normalization_type="LayerNorm",
              dense_act_fct="ELU",dense_n_layers=0,dense_dropout=0.0,dense_bias=False,dense_parametrizations="None", dense_residual_type="ResidualGate",dense_normalization_type="LayerNorm",
              rec_act_fct="Tanh",rec_n_layers=0,rec_type="GRU",rec_bidirectional=False,rec_dropout=0.0,rec_bias=False,rec_parametrizations="None", rec_residual_type="ResidualGate",rec_normalization_type="LayerNorm",
              tf_act_fct="ELU",tf_dense_dim=50,tf_n_layers=0,tf_dropout_rate_1=0.0,tf_dropout_rate_2=0.0,tf_attention_type="MultiHead",tf_positional_type ="absolute",tf_num_heads=1,tf_bias=False,tf_parametrizations="None",tf_residual_type="ResidualGate",tf_normalization_type="LayerNorm", tf_normalization_position="pre",
              merge_attention_type="MultiHead",merge_num_heads=1,merge_normalization_type="LayerNorm",merge_pooling_features=2,merge_pooling_type="MinMax",
              cls_pooling_features=2,cls_pooling_type="MinMax",
              metric_type="Euclidean",
              device=None, dtype=None):
    super().__init__()

    n_target_levels=len(target_levels)

    if core_net_type=="sequential":
      self.core_net=TEClassifierSequential(
        times=times, 
        features=features, 
        pad_value=pad_value,
        n_target_levels=n_target_levels,
        inc_cls_head=False,
        skip_connection_type=skip_connection_type,
        cls_type="regular",
        cls_pooling_type=cls_pooling_type,
        cls_pooling_features=cls_pooling_features, 
        feat_act_fct=feat_act_fct,
        feat_size=feat_size,
        feat_bias=feat_bias,
        feat_dropout=feat_dropout,
        feat_parametrizations=feat_parametrizations,
        feat_normalization_type=feat_normalization_type,
        ng_conv_act_fct=ng_conv_act_fct,
        ng_conv_n_layers=ng_conv_n_layers,
        ng_conv_ks_min=ng_conv_ks_min, 
        ng_conv_ks_max=ng_conv_ks_max, 
        ng_conv_bias=ng_conv_bias, 
        ng_conv_dropout=ng_conv_dropout,
        ng_conv_residual_type=ng_conv_residual_type,
        ng_conv_parametrizations=ng_conv_parametrizations,
        ng_conv_normalization_type=ng_conv_normalization_type,
        dense_act_fct=dense_act_fct,
        dense_n_layers=dense_n_layers,
        dense_dropout=dense_dropout,
        dense_bias=dense_bias,
        dense_parametrizations=dense_parametrizations,
        dense_residual_type=dense_residual_type,
        dense_normalization_type=dense_normalization_type,
        rec_act_fct=rec_act_fct,
        rec_n_layers=rec_n_layers,
        rec_type=rec_type,
        rec_bidirectional=rec_bidirectional,
        rec_dropout=rec_dropout,
        rec_bias=rec_bias,
        rec_parametrizations=rec_parametrizations,
        rec_residual_type=rec_residual_type,
        rec_normalization_type=rec_normalization_type,
        tf_act_fct=tf_act_fct,
        tf_dense_dim=tf_dense_dim,
        tf_n_layers=tf_n_layers,
        tf_dropout_rate_1=tf_dropout_rate_1,
        tf_dropout_rate_2=tf_dropout_rate_2,
        tf_attention_type=tf_attention_type,
        tf_positional_type =tf_positional_type ,
        tf_num_heads=tf_num_heads,
        tf_bias=tf_bias,
        tf_parametrizations=tf_parametrizations,
        tf_residual_type=tf_residual_type,
        tf_normalization_type=tf_normalization_type,
        tf_normalization_position=tf_normalization_position,
        device=device, 
        dtype=dtype
      )
    elif core_net_type=="parallel":
      self.core_net=TEClassifierParallel(
        times=times, 
        features=features, 
        pad_value=pad_value,
        n_target_levels=n_target_levels,
        inc_cls_head=False,
        merge_pooling_type=merge_pooling_type,
        merge_pooling_features=merge_pooling_features,
        shared_feat_layer=shared_feat_layer,
        feat_act_fct=feat_act_fct,
        feat_size=feat_size,
        feat_bias=feat_bias,
        feat_dropout=feat_dropout,
        feat_parametrizations=feat_parametrizations,
        feat_normalization_type=feat_normalization_type,
        ng_conv_act_fct=ng_conv_act_fct,
        ng_conv_n_layers=ng_conv_n_layers,
        ng_conv_ks_min=ng_conv_ks_min, 
        ng_conv_ks_max=ng_conv_ks_max, 
        ng_conv_bias=ng_conv_bias,
        ng_conv_dropout=ng_conv_dropout,
        ng_conv_residual_type=ng_conv_residual_type,
        ng_conv_parametrizations=ng_conv_parametrizations,
        ng_conv_normalization_type=ng_conv_normalization_type,
        dense_act_fct=dense_act_fct,
        dense_n_layers=dense_n_layers,
        dense_dropout=dense_dropout,
        dense_bias=dense_bias,
        dense_parametrizations=dense_parametrizations,
        dense_residual_type=dense_residual_type,
        dense_normalization_type=dense_normalization_type,
        rec_act_fct=rec_act_fct,
        rec_n_layers=rec_n_layers,
        rec_type=rec_type,
        rec_bidirectional=rec_bidirectional,
        rec_dropout=rec_dropout,
        rec_bias=rec_bias,
        rec_parametrizations=rec_parametrizations,
        rec_residual_type=rec_residual_type,
        rec_normalization_type=rec_normalization_type,
        tf_act_fct=tf_act_fct,
        tf_dense_dim=tf_dense_dim,
        tf_n_layers=tf_n_layers,
        tf_dropout_rate_1=tf_dropout_rate_1,
        tf_dropout_rate_2=tf_dropout_rate_2,
        tf_attention_type=tf_attention_type,
        tf_positional_type =tf_positional_type ,
        tf_num_heads=tf_num_heads,
        tf_bias=tf_bias,
        tf_parametrizations=tf_parametrizations,
        tf_residual_type=tf_residual_type,
        tf_normalization_type=tf_normalization_type,
        tf_normalization_position=tf_normalization_position,
        device=device, 
        dtype=dtype,
        merge_attention_type=merge_attention_type,
        merge_num_heads=merge_num_heads,
        merge_normalization_type=merge_normalization_type
      )
      
    self.embedding_dim=embedding_dim
    self.classes=torch.from_numpy(np.copy(target_levels))
    self.n_classes=n_target_levels
    
    self.trained_prototypes=torch.ones(1)
    self.class_labels=torch.ones(1)
    self.projection_type=projection_type
    
    if self.projection_type=="Regular":
      if core_net_type=="sequential":
        self.embedding_head=torch.nn.Linear(
          in_features=cls_pooling_features,
          out_features=self.embedding_dim,
          bias=True)
      elif core_net_type=="parallel":
        self.embedding_head=torch.nn.Linear(
          in_features=merge_pooling_features,
          out_features=self.embedding_dim,
          bias=True)
    elif self.projection_type=="PairwiseOrthogonal":
      if core_net_type=="sequential":
        self.embedding_head=pairwise_orthogonal_dense(
          input_size=cls_pooling_features,
          output_size=self.embedding_dim,
          pre_dense=False,
          bias=True)
      elif core_net_type=="parallel":
        self.embedding_head=pairwise_orthogonal_dense(
          input_size=merge_pooling_features,
          output_size=self.embedding_dim,
          pre_dense=False,
          bias=True)
    elif self.projection_type=="PairwiseOrthogonalDense":
      if core_net_type=="sequential":
        self.embedding_head=pairwise_orthogonal_dense(
          input_size=cls_pooling_features,
          output_size=self.embedding_dim,
          pre_dense=True,
          bias=True)
      elif core_net_type=="parallel":
        self.embedding_head=pairwise_orthogonal_dense(
          input_size=merge_pooling_features,
          output_size=self.embedding_dim,
          pre_dense=True,
          bias=True)          
    
    self.class_mean=layer_class_mean()
    self.metric=layer_protonet_metric(metric_type=metric_type)
    
  def forward(self, input_q,classes_q=None,input_s=None,classes_s=None, prediction_mode=True):
    #Sample set
    if input_s is None or classes_s is None:
      prototypes=self.trained_prototypes
      class_labels=self.class_labels
    else:
      #Get class labels in the samples which are also used for the query set
      class_labels=torch.unique(classes_s,sorted=True)
      n_classes=class_labels.size()[0]
      
      #Recode class labels in order to start at 0 for sample and query
      sample_classes=self.recode_classes(classes_s,class_labels)

      #Calculate Prototypes      
      prototypes=self.calc_prototypes(input_s=input_s,classes=sample_classes,total_classes=n_classes)

    #Query set
    query_embeddings=self.embed(input_q)

    #Calc distance from query embeddings to global prototypes
    distances=self.metric(x=query_embeddings,prototypes=prototypes)
    probabilities=torch.nn.Softmax(dim=1)(torch.exp(-distances))
      
    if prediction_mode==False:
      if classes_q==None:
        query_classes=None
      else:
        query_classes=self.recode_classes(classes_q,class_labels)
      return probabilities, distances, query_classes, query_embeddings, prototypes
    else:
      return probabilities
    
  def recode_classes(self,class_vector,class_labels):
    n_labels=class_labels.size()[0]
    new_classes=class_vector.clone()
    for index in range(n_labels):
      con=(new_classes==class_labels[index])
      new_classes=torch.where(condition=con,input=torch.tensor(index),other=new_classes)
    return new_classes
  
  def get_distances(self,inputs,prototypes):
    distances=self.metric(x=self.embed(inputs),prototypes=prototypes)
    return distances
  
  def get_metric_scale_factor(self):
    return self.metric.get_scaling_factor()
  
  def embed(self,inputs):
    embeddings=self.core_net(inputs)
    embeddings=self.embedding_head(embeddings)
    return embeddings
  
  def calc_prototypes(self,input_s,classes,total_classes):
    prototypes=self.embed(input_s)
    prototypes=self.class_mean(x=prototypes,classes=classes,total_classes=total_classes)
    return prototypes
  
  def set_trained_prototypes(self,prototypes,class_lables):
    self.trained_prototypes=torch.nn.Parameter(prototypes)
    self.class_labels=class_lables
    
  def get_trained_prototypes(self):
    return self.trained_prototypes
  
  def get_trained_class_labels(self):
    return self.class_labels
  
  def get_embedding_dim(self):
    return self.embedding_dim
