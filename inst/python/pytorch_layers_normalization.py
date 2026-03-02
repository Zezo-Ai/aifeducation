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
import torch.nn as nn
from torch.autograd import Function
import numpy as np
import math
import safetensors

#LayerNorm_with_Mask------------------------------------------------------------
#Layer generating the Layer Norm for sequential data.
# Returns a list with the following tensors
# * Input tensor
# * Sequence length of the tensors shape (Batch)
# * mask_times Mask on the level of complete sequences shape (Batch, Times)
# * mask_features Mask on the level of single features shape (Bath, Times, Features)
# True indicates that the sequence or feature is padded. If True these values should not be part 
# of further computations
# Layer Norm is applied to the last dimensio as described in the paper
# Layer Normalization in equation 4.
class LayerNorm_with_Mask(torch.nn.Module):
    def __init__(self, times, features,pad_value,eps=1e-5):
      super().__init__()
      self.eps=eps
      self.times=times
      self.features = features
      if isinstance(pad_value, torch.Tensor):
        self.pad_value=pad_value.detach()
      else:
        self.pad_value=torch.tensor(pad_value)
      self.gamma = torch.nn.Parameter(torch.ones(1, 1, self.features))

    def forward(self, x,mask_times):
      #Set padding value to zero for correct sum
      mask_features=get_FeatureMask_from_mask(mask_times,x.size(2))
      x_zeros=x*(~mask_features)
      #Calculate mean 
      #Create the sum for every timestep and case. These sum has the
      #shape (Batch, Times)
      mean=torch.sum(x_zeros,dim=2)/self.features
      
      #Calculate variance
      #Reshape mean to allow substraction shape (Batch, Times, Features)
      mean_long=torch.unsqueeze(mean,dim=2)
      mean_long=mean_long.expand(-1,-1,self.features)
      
      #Calculate variance which has shape (Batch, Times)
      var=torch.sum(torch.square((x_zeros-mean_long)),dim=2)/self.features
      var=torch.sqrt(var+self.eps)
      
      var_long=torch.unsqueeze(var,dim=2)
      var_long=var_long.expand(-1,-1,self.features)

      #Calculate normalized output
      gamma_long=self.gamma.expand(x.size(0),self.times,-1)
      normalized=gamma_long*(x_zeros-mean_long)/var_long
      
      #Insert padding values
      normalized=normalized.masked_fill(mask=mask_features,value=self.pad_value)

      return normalized, mask_times

# BatchNorm_with_Mask------------------------------------------------------------
# Layer generating the Batch Norm for sequential data.
# Returns a list with the following tensors
# * Input and output tensor (Batch, Times, Features) or (Batch, Features)
# * Sequence length of the tensors shape (Batch)
# * mask_times Mask on the level of complete sequences shape (Batch, Times)
# * mask_features Mask on the level of single features shape (Bath, Times, Features)
# True indicates that the sequence or feature is padded. If True these values should not be part
# of further computations
class BatchNorm_with_Mask(torch.nn.Module):
    def __init__(
        self,
        features: int,
        pad_value: torch.LongTensor | int,
        eps: float = 1e-5,
        alpha: float = 0.1,
    ):
        super().__init__()
        self.eps = eps
        self.alpha = alpha
        self.features = features
        if isinstance(pad_value, torch.Tensor):
            self.pad_value = pad_value.detach()
        else:
            self.pad_value = torch.tensor(pad_value)
        self.gamma = torch.nn.Parameter(torch.ones(1, 1, self.features))
        self.beta = torch.nn.Parameter(torch.zeros(1, 1, self.features))
        self.register_buffer("running_mean", torch.zeros((1, 1, self.features)))
        self.register_buffer("running_variance", torch.ones((1, 1, self.features)))

    def forward(
        self,
        x: torch.FloatTensor,  # (B, F) | (B, T, F)
        mask_times: torch.BoolTensor = None,
    ) -> tuple:
        # Ensure Format (Batch,Times,Features)
        if x.dim() == 2:
            x_reshaped = torch.unsqueeze(x, dim=1)
            if mask_times is None:
                mask_times = torch.zeros(
                    (x.size(0), 1), dtype=torch.bool, device=x.device
                )
        else:
            x_reshaped = x
            if mask_times is None:
                mask_times = torch.zeros(
                    (x.size(0), x.size(1)), dtype=torch.bool, device=x.device
                )
        B, T, F = x_reshaped.shape
        mask_features = get_FeatureMask_from_mask(mask_times, self.features)
        x_reshaped = (~mask_features) * x_reshaped
        gamma_expanded = self.gamma.expand(B, T, F)
        beta_expanded = self.beta.expand(B, T, F)

        if self.training == True:
            # calc batch mean and variance
            # (F_out), (F_out), B * T
            batch_mean, batch_variance, n_elements = self.calc_batch_statistics(
                x_reshaped, mask_times
            )
            if batch_mean is not None:
                # Update running mean and variance
                self.running_mean = (
                    1 - self.alpha
                ) * self.running_mean + self.alpha * torch.unsqueeze(
                    torch.unsqueeze(batch_mean.detach(), dim=0), dim=0
                )  # (1, 1, F_out)
                self.running_variance = (
                    1 - self.alpha
                ) * self.running_variance + self.alpha * (
                    n_elements / (n_elements - 1)
                ) * torch.unsqueeze(
                    torch.unsqueeze(batch_variance.detach(), dim=0), dim=0
                )  # (1, 1, F_out)
                # Normalize Scale and shift
                # self.eps in torch.sqrt is necessary for numeric stability
                y = (
                    gamma_expanded
                    * (x_reshaped - batch_mean)
                    / (torch.sqrt(batch_variance + self.eps) + self.eps)
                    + beta_expanded
                )  # (B, T, F)
            else:
                # Normalize Scale and shift
                y = (
                    gamma_expanded
                    * (x_reshaped - self.running_mean)
                    / (torch.sqrt(self.running_variance) + self.eps)
                    + beta_expanded
                )
        else:
            # Normalize Scale and shift
            y = (
                gamma_expanded
                * (x_reshaped - self.running_mean)
                / (torch.sqrt(self.running_variance) + self.eps)
                + beta_expanded
            )
        # Insert padding values
        # (B, T, F)
        normalized = y.masked_fill(mask=mask_features, value=self.pad_value)
        if x.dim() == 2:
            normalized = torch.squeeze(normalized, dim=1)  # (B, F)
        # Return results
        return normalized, mask_times  # (B, F) | (B, T, F); (B, T)

    def calc_batch_statistics(
        self,
        x: torch.FloatTensor,  # (B, T, F)
        mask_times: torch.BoolTensor,  # (B, T)
    ) -> tuple:
        # Transfrom to shape (Batch*Times,Feature)
        if x.dim() == 3:
            B, T, F = x.shape
            x_stacked = torch.reshape(x, shape=(B * T, F))
            mask_stacked = torch.reshape(
                mask_times, shape=(mask_times.size(0) * mask_times.size(1), 1)
            )  # (B * T, 1)
            mask_stacked = torch.squeeze(mask_stacked, dim=1)  # (B * T)
        else:
            x_stacked = x
            mask_stacked = mask_times
        # Select only the rows that are not masked
        x_sub = torch.index_select(
            input=x_stacked,  # (B * T, F)
            dim=0,
            index=torch.masked_select(
                input=torch.arange(start=0, end=x_stacked.size(0)).to(
                    mask_stacked.device
                ),
                mask=~mask_stacked,  # (B * T)
            ),
        )  # (B * T, F_out)
        # Calc meand and variance only if at least two rows exist
        if x_sub.size(0) >= 2:
            batch_mean = torch.mean(input=x_sub, dim=0)  # (F_out)
            batch_variance = torch.var(input=x_sub, dim=0)  # (F_out)
            batch_variance = torch.clamp(batch_variance, min=0.0, max=None)  # (F_out)
        else:
            batch_mean = None
            batch_variance = None
        n_elements = x_sub.size(0)  # B * T
        return batch_mean, batch_variance, n_elements  # (F_out), (F_out), B * T


#RMSNorm with mask--------------------------------------------------------------
class RMSNorm_with_Mask(nn.Module):
    def __init__(
        self,
        features: int,
        pad_value: torch.LongTensor | int,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(features))  # Multiplied
        self.features = features
        if isinstance(pad_value, torch.Tensor):
            self.pad_value = pad_value.detach()
        else:
            self.pad_value = torch.tensor(pad_value)

    def forward(
        self,
        x: torch.FloatTensor,  # (B, T, F)
        mask_times: torch.BoolTensor,  # (B, T)
    ) -> tuple:
        """
        x: (..., features)
        """
        mask_features = get_FeatureMask_from_mask(mask_times, self.features)
        rms = x * (~mask_features)
        rms = torch.pow(rms, 2)
        rms = torch.sum(rms, dim=2, keepdim=True) / self.features
        rms = torch.sqrt(rms + self.eps)  # eps for numeric stability
        x_norm = x / (rms + self.eps)
        x_norm = x_norm * self.gamma
        x_norm = x_norm.masked_fill(mask=mask_features, value=self.pad_value)
        return x_norm, mask_times

# PowerNorm with mask-----------------------------------------------------------
class PowerNormFunction(Function):
    @staticmethod
    def forward(ctx, X, gamma, beta, running_psi, nu, alpha, eps, training):
        if training:
            psi_B2 = (X**2).mean(dim=0).detach()  # mini-batch statistics
            running_psi.mul_(alpha).add_((1 - alpha) * psi_B2)
            psi = torch.sqrt(psi_B2 + eps)  # running statistics
        else:
            psi = torch.sqrt(running_psi + eps)  # running statistics

        X_hat = X / psi  # normalize
        Y = gamma * X_hat + beta  # scale and shift

        # save for backward
        ctx.save_for_backward(X_hat, psi, gamma)
        ctx.nu = nu
        ctx.alpha = alpha
        ctx.training = training

        return Y

    @staticmethod
    def backward(ctx, dL_dY):
        X_hat, psi, gamma = ctx.saved_tensors
        nu = ctx.nu
        alpha = ctx.alpha
        training = ctx.training

        dL_dX_hat = dL_dY * gamma  # intermediate gradient

        if training:
            # expectation over batch
            Gamma = (X_hat**2).mean(dim=0).detach()
            Lambda = (X_hat * dL_dX_hat).mean(dim=0).detach()

            # EMA (Exponential Moving Average) update of nu
            nu.mul_((1 - (1 - alpha) * Gamma)).add_((1 - alpha) * Lambda)

            correction = nu
        else:
            correction = 0.0

        dL_dX = (dL_dX_hat - X_hat * correction) / psi  # gradient x

        # gradients of params
        dL_dgamma = (dL_dY * X_hat).sum(dim=0)
        dL_dbeta = dL_dY.sum(dim=0)

        return dL_dX, dL_dgamma, dL_dbeta, None, None, None, None, None


class PowerNorm_with_Mask(nn.Module):
    def __init__(
        self,
        features: int,
        pad_value: torch.LongTensor = -100,
        alpha: float = 0.9,
        eps: float = 1e-6,
    ):
        super().__init__()

        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))

        self.register_buffer("running_psi", torch.ones(features))
        self.register_buffer("nu", torch.zeros(features))

        self.alpha = alpha
        self.eps = eps

        if isinstance(pad_value, torch.Tensor):
            self.pad_value = pad_value.detach()
        else:
            self.pad_value = torch.tensor(pad_value)

    def forward(
        self,
        x: torch.FloatTensor,
        mask_times: torch.BoolTensor = None,
    ) -> tuple:
        """
        x: (B, t, d) or (B, d)
        """

        if mask_times is None:
            mask_features = None
        else:
            mask_features = mask_features = get_FeatureMask_from_mask(
                mask_times, self.gamma.size(0)
            )

        if mask_features is not None:
            # Set padding value to zero
            x = x.masked_fill(mask_features, 1e-6)

        orig_shape = x.shape
        if x.dim() == 3:
            # combine batch and time to normalize by last dimension
            b, t, d = x.shape
            x = x.view(-1, d)

        x_norm = PowerNormFunction.apply(
            x,
            self.gamma,
            self.beta,
            self.running_psi,
            self.nu,
            self.alpha,
            self.eps,
            self.training,
        )

        if len(orig_shape) == 3:
            # return to the original shape
            x_norm = x_norm.view(orig_shape)

        if mask_features is not None:
            # Insert padding values
            x_norm = torch.where(
                condition=mask_features, input=self.pad_value, other=x_norm
            )

        return x_norm, mask_times


def get_layer_normalization(name,times, features,pad_value,eps=1e-6):
  if name=="LayerNorm":
    return LayerNorm_with_Mask(times=times,features=features,pad_value=pad_value,eps=eps)
  elif name=="BatchNorm":
    return BatchNorm_with_Mask(features=features,pad_value=pad_value,eps=eps,alpha=0.1)
  elif name=="RMSNorm":
    return RMSNorm_with_Mask(features=features,pad_value=pad_value,eps=eps)
  elif name=="PowerNorm":
    return PowerNorm_with_Mask(features=features,pad_value=pad_value,eps=eps,alpha=0.9)
  elif name=="None":
    return identity_layer(pad_value=pad_value,apply_masking=True)
