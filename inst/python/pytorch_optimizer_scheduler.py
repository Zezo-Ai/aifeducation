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
import math

def get_Optimizer(optimizer_method,params,lr_rate):
  if optimizer_method=="Adam":
    optimizer=torch.optim.Adam(lr=lr_rate,params=params,weight_decay=1e-3)
  elif optimizer_method=="RMSprop":
    optimizer=torch.optim.RMSprop(lr=lr_rate,params=params,momentum=0.90)
  elif optimizer_method=="AdamW":
    optimizer=torch.optim.AdamW(lr=lr_rate,params=params)
  elif optimizer_method=="SGD":
    optimizer=torch.optim.SGD(params=params, lr=lr_rate, momentum=0.90, dampening=0, weight_decay=0, nesterov=False, maximize=False, foreach=None, differentiable=False, fused=None)
  return optimizer

def get_lr_scheduler(optimizer,scheduler_type,lr_warm_up_ratio,total_epochs,batches_per_epoch,max_lr,min_lr):
  warm_up_steps=math.floor(total_epochs*lr_warm_up_ratio)
  main_steps=total_epochs-warm_up_steps
  if scheduler_type=="Linear":
    scheduler_main=torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1,end_factor=min_lr/max_lr, total_iters=main_steps)
  elif scheduler_type=="Cyclic":
    scheduler_main=torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=min_lr, max_lr=max_lr, step_size_up=5*batches_per_epoch)
  elif scheduler_type=="None":
    return None
  
  if lr_warm_up_ratio>0.0 and scheduler_type!="None":
    scheduler_warm_up = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=min_lr/max_lr,end_factor=1, total_iters=warm_up_steps)
    scheduler = torch.optim.lr_scheduler.SequentialLR(schedulers = [scheduler_warm_up, scheduler_main], optimizer=optimizer,milestones=[warm_up_steps])
  else:
    scheduler=scheduler_main
  return scheduler
