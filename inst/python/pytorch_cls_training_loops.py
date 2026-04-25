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

#Functions that are part of the training loop
def get_device():
  return 'cuda' if torch.cuda.is_available() else 'cpu'

def get_dtype(device):
  if device=="cpu":
    current_dtype=torch.float
  else:
    current_dtype=torch.float
    
def get_loss_cls_fct(name,class_weights):
  if name =="CrossEntropyLoss":
    loss_fct=torch.nn.CrossEntropyLoss(
        reduction="none",
        weight = class_weights)
  elif name =="FocalLoss":
    loss_fct=focal_loss(
      gamma=2,
      class_weights = class_weights
    )
  return loss_fct

def get_loss_cls_pt_fct(name,margin,alpha):
  if name=="MultiWayContrastiveLoss":
    fct=multi_way_contrastive_loss(
      alpha=alpha,
      margin=margin)
  elif name=="MultiWayContrastiveLossFC":
    fct=multi_way_contrastive_loss_fc(
      alpha=alpha,
      margin=margin)
  return fct

def build_data_loaders(train_data, val_data, batch_size, test_data=None, pin_memory=False):
  trainloader=torch.utils.data.DataLoader(
    train_data,
    batch_size=batch_size,
    pin_memory=pin_memory,
    shuffle=True)
  valloader=torch.utils.data.DataLoader(
    val_data,
    batch_size=batch_size,
    pin_memory=pin_memory,
    shuffle=False)
  if not (test_data is None):
    testloader=torch.utils.data.DataLoader(
      test_data,
      batch_size=batch_size,
      pin_memory=pin_memory,
      shuffle=False)
  else:
    testloader=None
  return trainloader, valloader, testloader

def create_metric_storage(metric_names,epochs,inc_test):
  storage={}
  for metric in metric_names:
    if inc_test:
      tmp_metric_storage=np.ones((3,epochs))*-100
    else:
      tmp_metric_storage=np.ones((2,epochs))*-100
    storage[metric]=  tmp_metric_storage
  storage["checkpoints"]=np.zeros((epochs))
  return storage

def calc_cls_performance_measures(confusion_matrix,n_classes):
  with torch.no_grad():
      acc=torch.sum(torch.diagonal(confusion_matrix))/torch.sum(confusion_matrix)
      bacc=torch.sum(torch.diagonal(confusion_matrix)/torch.sum(confusion_matrix,dim=1))/n_classes
      avg_iota=torch.diagonal(confusion_matrix)/(torch.sum(confusion_matrix,dim=0)+torch.sum(confusion_matrix,dim=1)-torch.diagonal(confusion_matrix))
      avg_iota=torch.sum(avg_iota)/n_classes
  return {"accuracy":acc, "balanced_accuracy":bacc, "avg_iota":avg_iota}

def add_metrics(metrics,storage,cblock,epoch):
  if cblock=="train":
    idx=0
  elif cblock=="val":
    idx=1
  elif cblock=="test":
    idx=2
  for key in metrics.keys():
    storage[key][idx,epoch]=metrics[key]

def print_epoch_results(trace,loss_only,metric_storage,epoch,epochs,metric_criterion,best_metric,best_loss):
  if trace:
    if (epoch+1)==epochs:
      end_string="\n"
    else:
      end_string="\r"
    if loss_only:
      loss=metric_storage["loss"]
      train_loss=loss[0,epoch]
      val_loss=loss[1,epoch]
      print("{:.4f} % | Train Loss {:.8f} | Val Loss {:.8f} Best {:.8f}".format(
              (epoch+1)/epochs,
              train_loss,
              val_loss,
              best_loss
              ),
            end=end_string
      )
    else:
      metric=metric_storage[metric_criterion]
      train_metric=metric[0,epoch]
      val_metric=metric[1,epoch]
      loss=metric_storage["loss"]
      train_loss=loss[0,epoch]
      val_loss=loss[1,epoch]
      print("{:.4f} % | Train Loss {:.4f} {} {:.4f} | Val Loss {:.4f} Best {:.4f} {} {:.4f} Best {:.4f}".format(
              (epoch+1)/epochs,
              train_loss,
              metric_criterion,
              train_metric,
              val_loss,
              best_loss,
              metric_criterion,
              val_metric,
              best_metric
              ),
            end=end_string
      )

def check_and_set_checkpoints_cls(use_callback,model,filepath,epoch,metric_storage,best_val_avg_iota,best_val_loss,best_acc,best_bacc,acc_val,bacc_val,avg_iota_val,val_loss):
  if use_callback==True:
      if (avg_iota_val>best_val_avg_iota) or (avg_iota_val==best_val_avg_iota and acc_val>best_acc) or (avg_iota_val==best_val_avg_iota and acc_val==best_acc and val_loss<best_val_loss):
        torch.save(model.state_dict(),filepath)
        best_bacc=bacc_val
        best_val_avg_iota=avg_iota_val
        best_acc=acc_val
        best_val_loss=val_loss
        metric_storage["checkpoints"][epoch]=1
  return best_val_loss, best_acc,best_bacc,best_val_avg_iota  


def run_epoch_cls(model,dataloader,loss_fct,optimizer,scaler,scheduler,amp,epoch,n_classes,device,current_dtype,cblock,metric_storage,logger):
  total_loss=0.0
  confusion_matrix=torch.zeros(size=(n_classes,n_classes),device=device,dtype=current_dtype)

  if cblock=="train":
    optimizer.zero_grad()
    model.train()
    ctx=torch.enable_grad()
  else:
    model.eval()
    ctx=torch.no_grad()

  for batch in dataloader:
    with ctx: 
      inputs=batch["input"]
      labels=batch["labels"]
      inputs = inputs.to(device,dtype=current_dtype)
      labels=labels.to(device,dtype=current_dtype)
      if "sample_weights" in batch.keys():
        sample_weights=batch["sample_weights"]
        sample_weights=torch.reshape(input=sample_weights,shape=(sample_weights.size(dim=0),1))
        sample_weights=sample_weights.to(device,dtype=current_dtype)
      else:
         sample_weights=torch.ones((inputs.size(0)),device=device,dtype=current_dtype)/inputs.size(0)
      
      if cblock=="train":
        optimizer.zero_grad()
      with torch.autocast(device_type=device, dtype=None, enabled=amp):  
        outputs=model(inputs,prediction_mode=False)
        loss=loss_fct(outputs,labels)*sample_weights
        loss=loss.mean()
      if cblock=="train":
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()  
        #loss.backward()
        #optimizer.step()
        if scheduler!=None:
          scheduler.step()
      #Metrics
      total_loss +=loss.item()
      label_idx=labels.max(dim=1).indices
    confusion_matrix+=multiclass_confusion_matrix(input=outputs,target=label_idx,num_classes=n_classes)
    
    #Update log file
    logger.inc_value("bottom")
    logger.write_log()
    logger.write_history_log(metric_storage["loss"])
  #Calc final metrics for epoch
  results=calc_cls_performance_measures(
    confusion_matrix=confusion_matrix,
    n_classes=n_classes
  )
  results.update({"loss":total_loss/len(dataloader)})
  #Save metrics
  add_metrics(
    metrics=results,
    storage=metric_storage,
    cblock=cblock,
    epoch=epoch
  )
  return results

def run_epoch_cls_pt(model,dataloader,loss_fct,optimizer,scaler, scheduler,amp,epoch,Ns,Nq,n_classes,device,current_dtype,cblock,metric_storage,logger):
  total_loss=0.0
  confusion_matrix=torch.zeros(size=(n_classes,n_classes),device=device,dtype=current_dtype)

  if cblock=="train":
    optimizer.zero_grad()
    model.train()
    ctx=torch.enable_grad()
  else:
    model.eval()
    ctx=torch.no_grad()

  for batch in dataloader:
    with ctx:
      inputs=batch["input"]
      labels=batch["labels"]
      if cblock=="train":
        sample_inputs=inputs[0:(n_classes*Ns)].clone()
        query_inputs=inputs[(n_classes*Ns):(n_classes*(Ns+Nq))].clone()
        sample_classes=labels[0:(n_classes*Ns)].clone()
        query_classes=labels[(n_classes*Ns):(n_classes*(Ns+Nq))].clone()
        sample_inputs = sample_inputs.to(device,dtype=current_dtype)
        query_inputs = query_inputs.to(device,dtype=current_dtype)
        sample_classes = sample_classes.to(device,dtype=current_dtype)
        query_classes = query_classes.to(device,dtype=current_dtype)

        optimizer.zero_grad()
        with torch.autocast(device_type=device, dtype=None, enabled=amp):
          outputs=model(
            input_q=query_inputs,
            classes_q=query_classes,
            input_s=sample_inputs,
            classes_s=sample_classes,
            prediction_mode=False
          )
          loss=loss_fct(
            classes_q=outputs[2],
            distance_matrix=outputs[1],
            metric_scale_factor=model.get_metric_scale_factor().detach()
          )
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()  
        #loss.backward()
        #optimizer.step()
        if scheduler!=None:
          scheduler.step()
        #Metrics
        total_loss +=loss.item()
        pred_idx=outputs[0].max(dim=1).indices.to(dtype=torch.long,device=device)
        label_idx=query_classes.to(dtype=torch.long,device=device)  
      else:
        inputs = inputs.to(device,dtype=current_dtype)
        labels=labels.to(device,dtype=current_dtype)
        with torch.autocast(device_type=device, dtype=None, enabled=amp):
          outputs=model(input_q=inputs,classes_q=labels,prediction_mode=False)
          loss=loss_fct(
            classes_q=outputs[2],
            distance_matrix=outputs[1],
            metric_scale_factor=model.get_metric_scale_factor().detach()
          )
        #Metrics
        total_loss +=loss.item()
        pred_idx=outputs[0].max(dim=1).indices.to(dtype=torch.long,device=device)
        label_idx=outputs[2].to(dtype=torch.long,device=device)
    confusion_matrix+=multiclass_confusion_matrix(input=pred_idx,target=label_idx,num_classes=n_classes)
    #Update log file
    logger.inc_value("bottom")
    logger.write_log()
    logger.write_history_log(metric_storage["loss"])
  #Calculate prototypes
  if cblock=="train":
    model.eval()
    class_mean_prototypes,class_label=calc_trained_prototypes_batch(
      n_classes=n_classes,
      model=model,
      data_loader=dataloader,
      device=device,
      dtype=current_dtype
      )
    model.set_trained_prototypes(
      prototypes=class_mean_prototypes,
      class_lables=class_label
      )
  #Calc final metrics for epoch
  results=calc_cls_performance_measures(
    confusion_matrix=confusion_matrix,
    n_classes=n_classes
  )
  results.update({"loss":total_loss/len(dataloader)})
  #Save metrics
  add_metrics(
    metrics=results,
    storage=metric_storage,
    cblock=cblock,
    epoch=epoch
  )
  return results

def TeClassifierTrain(model,loss_cls_fct_name , optimizer_method,scheduler_type,amp, lr_rate,lr_min, lr_warm_up_ratio, epochs, trace,batch_size,
train_data,val_data,filepath,use_callback,n_classes,class_weights,test_data=None,
log_dir=None, log_write_interval=10, log_top_value=0, log_top_total=1, log_top_message="NA"):
  #Prepare model
  device=get_device()
  current_dtype=get_dtype(device)
  model.to(device=device,dtype=current_dtype)
  #Prepare loss function
  class_weights=class_weights.clone()
  class_weights=class_weights.to(device)
  loss_fct=get_loss_cls_fct(name=loss_cls_fct_name,class_weights=class_weights)
  loss_fct.to(device=device,dtype=current_dtype)
  #create data loader
  trainloader, valloader, testloader = build_data_loaders(
    train_data=train_data,
    val_data=val_data,
    test_data=test_data,
    batch_size=batch_size,
    pin_memory = True if device=="cuda" else False
  )
  #Create optimizer and scheduler
  optimizer=get_Optimizer(
    optimizer_method,
    params=model.parameters(),
    lr_rate=lr_rate
  )
  scheduler=get_lr_scheduler(
    optimizer=optimizer,
    scheduler_type=scheduler_type,
    lr_warm_up_ratio=lr_warm_up_ratio,
    total_epochs=epochs,
    batches_per_epoch=len(trainloader),
    max_lr=lr_rate,
    min_lr=lr_min
  )
  amp_scaler=torch.amp.GradScaler(device ,enabled=amp)
  #Numpys for Saving Training History
  metric_storage=create_metric_storage(
    metric_names=["loss","accuracy","balanced_accuracy","avg_iota"],
    epochs=epochs,
    inc_test=True if not (test_data is None) else False
  )
  # Init checkpoint values
  best_bacc=float('-inf')
  best_acc=float('-inf')
  best_val_loss=float('inf')
  best_val_avg_iota=float('-inf')
  #Logger
  total_steps=len(trainloader)+len(valloader)
  if not (test_data is None):
    total_steps=total_steps+len(testloader)
  logger=LogWriter(
    log_file=log_dir+"/aifeducation_state.log" if not (log_dir is None) else None,
    log_file_loss =log_dir+"/aifeducation_loss.log" if not (log_dir is None) else None,
    value_top = log_top_value, 
    value_middle = 0, 
    value_bottom = 0,
    total_top = log_top_total, 
    total_middle = epochs, 
    total_bottom = total_steps, 
    message_top = log_top_message, 
    message_middle = "Epoch",
    message_bottom = "Steps",
    last_log = None, 
    write_interval = log_write_interval
  )
  # Start loop    
  for epoch in range(epochs):
    train_results=run_epoch_cls(
      model=model,
      dataloader=trainloader,
      optimizer=optimizer,
      scaler=amp_scaler,
      scheduler=scheduler,
      amp=amp,
      loss_fct=loss_fct,
      epoch=epoch,
      n_classes=n_classes,
      device=device,
      current_dtype=current_dtype,
      cblock="train",
      metric_storage=metric_storage,
      logger=logger
    )
    val_results=run_epoch_cls(
      model=model,
      dataloader=valloader,
      loss_fct=loss_fct,
      optimizer=optimizer,
      scaler=amp_scaler,
      scheduler=scheduler,
      amp=amp,
      epoch=epoch,
      n_classes=n_classes,
      device=device,
      current_dtype=current_dtype,
      cblock="val",
      metric_storage=metric_storage,
      logger=logger
    )
    if testloader is not None:
      test_results=run_epoch_cls(
        model=model,
        dataloader=testloader,
        optimizer=optimizer,
        scaler=amp_scaler,
        scheduler=scheduler,
        amp=amp,
        loss_fct=loss_fct,
        epoch=epoch,
        n_classes=n_classes,
        device=device,
        current_dtype=current_dtype,
        cblock="test",
        metric_storage=metric_storage,
        logger=logger
      )
    #Update logger   
    logger.reset_value(level="bottom")
    logger.inc_value(level="middle")
    #Callback-------------------------------------------------------------------
    best_val_loss, best_acc, best_bacc, best_val_avg_iota = check_and_set_checkpoints_cls(
      use_callback=use_callback,
      model=model,
      filepath=filepath,
      epoch=epoch,
      metric_storage=metric_storage,
      best_val_avg_iota=best_val_avg_iota,
      best_val_loss=best_val_loss,
      best_acc=best_acc,
      best_bacc=best_bacc,
      acc_val=val_results["accuracy"],
      bacc_val=val_results["balanced_accuracy"],
      avg_iota_val=val_results["avg_iota"],
      val_loss=val_results["loss"]
    )
    #Trace---------------------------------------------------------------------
    print_epoch_results(
      trace=trace,
      loss_only=False,
      metric_storage=metric_storage,
      epoch=epoch,
      epochs=epochs,
      metric_criterion="avg_iota",
      best_metric=best_val_avg_iota,
      best_loss=best_val_loss
    )
    #Check if there are furhter information for training-----------------------
    # If there are no addtiononal information. Stop training and continue
    if train_results["loss"]<1e-3 and train_results["accuracy"]==1 and train_results["balanced_accuracy"]==1 and train_results["avg_iota"]==1:
      break
  #Finalize--------------------------------------------------------------------
  if use_callback==True:
    model.load_state_dict(torch.load(filepath,weights_only=True))
  return metric_storage

def TeClassifierTrainPrototype(model,loss_pt_fct_name , optimizer_method, scheduler_type, amp,lr_rate,lr_min, lr_warm_up_ratio, epochs, trace,Ns,Nq,
loss_alpha, loss_margin, train_data,val_data,filepath,use_callback,n_classes,sampling_separate,sampling_shuffle,test_data=None,
log_dir=None, log_write_interval=10, log_top_value=0, log_top_total=1, log_top_message="NA"):
  #Prepare model
  device=get_device()
  current_dtype=get_dtype(device)
  model.to(device=device,dtype=current_dtype)
  #Prepare loss function
  loss_fct=get_loss_cls_pt_fct(
    name=loss_pt_fct_name,
    alpha=loss_alpha,
    margin=loss_margin
  )
  #Numpys for Saving Training History
  metric_storage=create_metric_storage(
    metric_names=["loss","accuracy","balanced_accuracy","avg_iota"],
    epochs=epochs,
    inc_test=True if not (test_data is None) else False
  )
  # Init checkpoint values
  best_bacc=float('-inf')
  best_acc=float('-inf')
  best_val_loss=float('inf')
  best_val_avg_iota=float('-inf')
  #Set Up Loaders
  ProtoNetSampler_Train=MetaLernerBatchSampler(
  targets=train_data["labels"][range(0,len(train_data))],
  Ns=Ns,
  Nq=Nq,
  separate=sampling_separate,
  shuffle=sampling_shuffle)
  trainloader=torch.utils.data.DataLoader(
    train_data,
    pin_memory = True if device=="cuda" else False,
    batch_sampler=ProtoNetSampler_Train)
  valloader=torch.utils.data.DataLoader(
    val_data,
    pin_memory = True if device=="cuda" else False,
    batch_size=Ns+Nq,
    shuffle=False)
  if not (test_data is None):
    testloader=torch.utils.data.DataLoader(
      test_data,
      pin_memory = True if device=="cuda" else False,
      batch_size=Ns+Nq,
      shuffle=False)
  else:
    testloader=None
  #Create optimizer and scheduler    
  optimizer=get_Optimizer(
    optimizer_method,
    params=model.parameters(),
    lr_rate=lr_rate
  )
  scheduler=get_lr_scheduler(
    optimizer=optimizer,
    scheduler_type=scheduler_type,
    lr_warm_up_ratio=lr_warm_up_ratio,
    total_epochs=epochs,
    batches_per_epoch=len(trainloader),
    max_lr=lr_rate,
    min_lr=lr_min
  )
  amp_scaler=torch.amp.GradScaler(device ,enabled=amp)
 #Logger
  total_steps=len(trainloader)+len(valloader)
  if not (test_data is None):
    total_steps=total_steps+len(testloader)
  logger=LogWriter(
    log_file=log_dir+"/aifeducation_state.log" if not (log_dir is None) else None,
    log_file_loss =log_dir+"/aifeducation_loss.log" if not (log_dir is None) else None,
    value_top = log_top_value, 
    value_middle = 0, 
    value_bottom = 0,
    total_top = log_top_total, 
    total_middle = epochs, 
    total_bottom = total_steps, 
    message_top = log_top_message, 
    message_middle = "Epoch",
    message_bottom = "Steps",
    last_log = None, 
    write_interval = log_write_interval
  )
  #Start loop
  for epoch in range(epochs):
    train_results=run_epoch_cls_pt(
      model=model,
      dataloader=trainloader,
      optimizer=optimizer,
      scaler=amp_scaler,
      scheduler=scheduler,
      amp=amp,
      loss_fct=loss_fct,
      epoch=epoch,
      Ns=Ns,
      Nq=Nq,
      n_classes=n_classes,
      device=device,
      current_dtype=current_dtype,
      cblock="train",
      metric_storage=metric_storage,
      logger=logger
    )
    val_results=run_epoch_cls_pt(
      model=model,
      dataloader=valloader,
      loss_fct=loss_fct,
      optimizer=optimizer,
      scaler=amp_scaler,
      scheduler=scheduler,
      amp=amp,
      epoch=epoch,
      Ns=Ns,
      Nq=Nq,
      n_classes=n_classes,
      device=device,
      current_dtype=current_dtype,
      cblock="val",
      metric_storage=metric_storage,
      logger=logger
    )
    if testloader is not None:
      test_results=run_epoch_cls_pt(
        model=model,
        dataloader=testloader,
        optimizer=optimizer,
        scaler=amp_scaler,
        scheduler=scheduler,
        amp=amp,
        loss_fct=loss_fct,
        epoch=epoch,
        Ns=Ns,
        Nq=Nq,
        n_classes=n_classes,
        device=device,
        current_dtype=current_dtype,
        cblock="test",
        metric_storage=metric_storage,
        logger=logger
      )    
    #Update logger   
    logger.reset_value(level="bottom")
    logger.inc_value(level="middle")
    #Callback-------------------------------------------------------------------
    best_val_loss, best_acc, best_bacc, best_val_avg_iota = check_and_set_checkpoints_cls(
      use_callback=use_callback,
      model=model,
      filepath=filepath,
      epoch=epoch,
      metric_storage=metric_storage,
      best_val_avg_iota=best_val_avg_iota,
      best_val_loss=best_val_loss,
      best_acc=best_acc,
      best_bacc=best_bacc,
      acc_val=val_results["accuracy"],
      bacc_val=val_results["balanced_accuracy"],
      avg_iota_val=val_results["avg_iota"],
      val_loss=val_results["loss"]
    )
    #Trace---------------------------------------------------------------------
    print_epoch_results(
      trace=trace,
      loss_only=False,
      metric_storage=metric_storage,
      epoch=epoch,
      epochs=epochs,
      metric_criterion="avg_iota",
      best_metric=best_val_avg_iota,
      best_loss=best_val_loss
    )
    #Check if there are furhter information for training-----------------------
    # If there are no addtiononal information. Stop training and continue
    if train_results["loss"]<1e-3 and train_results["accuracy"]==1 and train_results["balanced_accuracy"]==1 and train_results["avg_iota"]==1:
      break
  #Finalize--------------------------------------------------------------------
  if use_callback==True:
    model.load_state_dict(torch.load(filepath,weights_only=True))
  return metric_storage


def calc_trained_prototypes_batch(n_classes,model,data_loader,device,dtype):
    model.eval()
    
    running_class_values=torch.zeros((n_classes,model.get_embedding_dim())).to(device)
    running_class_freq=torch.zeros(n_classes).to(device)
    
    for batch in data_loader:
      #assign colums of the batch
      inputs=batch["input"]
      labels=batch["labels"]
      
      inputs = inputs.to(device,dtype=dtype)
      labels=labels.to(device,dtype=dtype)
      labels_one_hot=torch.nn.functional.one_hot(labels.to(dtype=torch.long),num_classes=n_classes)

      embeddings=model.embed(inputs).to(device)

      running_class_values=running_class_values+torch.matmul(
        torch.transpose(labels_one_hot.to(dtype=embeddings.dtype),dim0=1,dim1=0),
        embeddings
      )
      running_class_freq=running_class_freq+torch.sum(labels_one_hot,dim=0)
      
    running_class_freq=torch.unsqueeze(running_class_freq,-1)
    running_class_freq=running_class_freq.repeat((1,model.get_embedding_dim()))
    
    class_mean_prototypes=running_class_values/running_class_freq
    
    class_labels=torch.arange(start=0, end=n_classes, step=1)
    return class_mean_prototypes, class_labels
