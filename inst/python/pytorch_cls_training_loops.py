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

def TeClassifierTrain(model,loss_cls_fct_name , optimizer_method,scheduler_type, lr_rate,lr_min, lr_warm_up_ratio, epochs, trace,batch_size,
train_data,val_data,filepath,use_callback,n_classes,class_weights,test_data=None,
log_dir=None, log_write_interval=10, log_top_value=0, log_top_total=1, log_top_message="NA"):
  
  device=('cuda' if torch.cuda.is_available() else 'cpu')
  
  if device=="cpu":
    #current_dtype=float
    current_dtype=torch.float64
    model.to(device,dtype=current_dtype)
  else:
    current_dtype=torch.double
    model.to(device,dtype=current_dtype)
  
  class_weights=class_weights.clone()
  class_weights=class_weights.to(device)
  
  if loss_cls_fct_name =="CrossEntropyLoss":
    loss_fct=torch.nn.CrossEntropyLoss(
        reduction="none",
        weight = class_weights)
  elif loss_cls_fct_name =="FocalLoss":
    loss_fct=focal_loss(
      gamma=2,
      class_weights = class_weights
    )
  
  trainloader=torch.utils.data.DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True)
    
  valloader=torch.utils.data.DataLoader(
    val_data,
    batch_size=batch_size,
    shuffle=False)
    
  if not (test_data is None):
    testloader=torch.utils.data.DataLoader(
      test_data,
      batch_size=batch_size,
      shuffle=False)
  
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
      
  #Tensor for Saving Training History
  if not (test_data is None):
    history_loss=torch.ones(size=(3,epochs),requires_grad=False)*-100
    history_acc=torch.ones(size=(3,epochs),requires_grad=False)*-100
    history_bacc=torch.ones(size=(3,epochs),requires_grad=False)*-100
    history_avg_iota=torch.ones(size=(3,epochs),requires_grad=False)*-100
  else:
    history_loss=torch.ones(size=(2,epochs),requires_grad=False)*-100
    history_acc=torch.ones(size=(2,epochs),requires_grad=False)*-100
    history_bacc=torch.ones(size=(2,epochs),requires_grad=False)*-100
    history_avg_iota=torch.ones(size=(2,epochs),requires_grad=False)*-100
  
  best_bacc=float('-inf')
  best_acc=float('-inf')
  best_val_loss=float('inf')
  best_val_avg_iota=float('-inf')
  
  #Log file
  if not (log_dir is None):
    log_file=log_dir+"/aifeducation_state.log"
    log_file_loss=log_dir+"/aifeducation_loss.log"
    last_log=None
    last_log_loss=None
    current_step=0
    total_epochs=epochs
    total_steps=len(trainloader)+len(valloader)
    if not (test_data is None):
      total_steps=total_steps+len(testloader)
    

  for epoch in range(epochs):
    #logging
    current_step=0
    
    #Training------------------------------------------------------------------
    train_loss=0.0
    n_matches_train=0
    n_total_train=0
    confusion_matrix_train=torch.zeros(size=(n_classes,n_classes))
    confusion_matrix_train=confusion_matrix_train.to(device,dtype=torch.double)
    
    model.train(True)
    torch.autograd.set_detect_anomaly(True)
    for batch in trainloader:
      inputs=batch["input"]
      labels=batch["labels"]

      sample_weights=batch["sample_weights"]
      sample_weights=torch.reshape(input=sample_weights,shape=(sample_weights.size(dim=0),1))

      inputs = inputs.to(device,dtype=current_dtype)
      labels=labels.to(device,dtype=current_dtype)
      sample_weights=sample_weights.to(device,dtype=current_dtype)
      
      optimizer.zero_grad()
      
      outputs=model(inputs,prediction_mode=False)
      loss=loss_fct(outputs,labels)*sample_weights
      loss=loss.mean()
      loss.backward()
      optimizer.step()
      train_loss +=loss.item()
      
      #Calc Accuracy
      pred_idx=torch.nn.Softmax(dim=1)(outputs).max(dim=1).indices
      label_idx=labels.max(dim=1).indices
          
      match=(pred_idx==label_idx)
      n_matches_train+=match.sum().item()
      n_total_train+=outputs.size(0)
      
      #Calc Balanced Accuracy
      confusion_matrix_train+=multiclass_confusion_matrix(input=outputs,target=label_idx,num_classes=n_classes)
      
      #Update log file
      if not (log_dir is None):
        current_step+=1
        last_log=write_log_py(log_file=log_file, value_top = log_top_value, value_middle = epoch+1, value_bottom = current_step,
                  total_top = log_top_total, total_middle = epochs, total_bottom = total_steps, message_top = log_top_message, message_middle = "Epochs",
                  message_bottom = "Steps", last_log = last_log, write_interval = log_write_interval)
        last_log_loss=write_log_performance_py(log_file=log_file_loss, history=history_loss.numpy().tolist(), last_log = last_log_loss, write_interval = log_write_interval)
     
    #Calc final metrics for epoch  
    acc_train=n_matches_train/n_total_train
    bacc_train=torch.sum(torch.diagonal(confusion_matrix_train)/torch.sum(confusion_matrix_train,dim=1))/n_classes
    avg_iota_train=torch.diagonal(confusion_matrix_train)/(torch.sum(confusion_matrix_train,dim=0)+torch.sum(confusion_matrix_train,dim=1)-torch.diagonal(confusion_matrix_train))
    avg_iota_train=torch.sum(avg_iota_train)/n_classes
    
    #Update learning rate
    if scheduler!=None:
      scheduler.step()
    #print(scheduler.get_last_lr())

    #Validation----------------------------------------------------------------
    val_loss=0.0
    n_matches_val=0
    n_total_val=0
    
    confusion_matrix_val=torch.zeros(size=(n_classes,n_classes))
    confusion_matrix_val=confusion_matrix_val.to(device,dtype=torch.double)
    
    model.eval()
    with torch.no_grad():
      for batch in valloader:
        inputs=batch["input"]
        labels=batch["labels"]

        inputs = inputs.to(device,dtype=current_dtype)
        labels=labels.to(device,dtype=current_dtype)
      
        outputs=model(inputs,prediction_mode=False)

        loss=loss_fct(outputs,labels).mean()
        val_loss +=loss.item()
      
        pred_idx=torch.nn.Softmax(dim=1)(outputs).max(dim=1).indices
        label_idx=labels.max(dim=1).indices
          
        match=(pred_idx==label_idx)
        n_matches_val+=match.sum().item()
        n_total_val+=outputs.size(0)
        
        #Calc Balanced Accuracy
        confusion_matrix_val+=multiclass_confusion_matrix(input=outputs,target=label_idx,num_classes=n_classes)
        
        #Update log file
        if not (log_dir is None):
          current_step+=1
          last_log=write_log_py(log_file=log_file, value_top = log_top_value, value_middle = epoch+1, value_bottom = current_step,
                    total_top = log_top_total, total_middle = epochs, total_bottom = total_steps, message_top = log_top_message, message_middle = "Epochs",
                    message_bottom = "Steps", last_log = last_log, write_interval = log_write_interval)
          last_log_loss=write_log_performance_py(log_file=log_file_loss, history=history_loss.numpy().tolist(), last_log = last_log_loss, write_interval = log_write_interval)

    acc_val=n_matches_val/n_total_val
    bacc_val=torch.sum(torch.diagonal(confusion_matrix_val)/torch.sum(confusion_matrix_val,dim=1))/n_classes
    avg_iota_val=torch.diagonal(confusion_matrix_val)/(torch.sum(confusion_matrix_val,dim=0)+torch.sum(confusion_matrix_val,dim=1)-torch.diagonal(confusion_matrix_val))
    avg_iota_val=torch.sum(avg_iota_val)/n_classes
    
    #Test----------------------------------------------------------------------
    if not (test_data is None):
      test_loss=0.0
      n_matches_test=0
      n_total_test=0
      
      confusion_matrix_test=torch.zeros(size=(n_classes,n_classes))
      confusion_matrix_test=confusion_matrix_test.to(device,dtype=torch.double)
  
      model.eval()
      with torch.no_grad():
        for batch in testloader:
          inputs=batch["input"]
          labels=batch["labels"]

          inputs = inputs.to(device,dtype=current_dtype)
          labels=labels.to(device,dtype=current_dtype)
        
          outputs=model(inputs,prediction_mode=False)
          
          loss=loss_fct(outputs,labels).mean()
          test_loss +=loss.item()
        
          pred_idx=torch.nn.Softmax(dim=1)(outputs).max(dim=1).indices
          label_idx=labels.max(dim=1).indices
            
          match=(pred_idx==label_idx)
          n_matches_test+=match.sum().item()
          n_total_test+=outputs.size(0)
          
          #Calc Balanced Accuracy
          confusion_matrix_test+=multiclass_confusion_matrix(input=outputs,target=label_idx,num_classes=n_classes)
          
          #Update log file
          if not (log_dir is None):
            current_step+=1
            last_log=write_log_py(log_file=log_file, value_top = log_top_value, value_middle = epoch+1, value_bottom = current_step,
                      total_top = log_top_total, total_middle = epochs, total_bottom = total_steps, message_top = log_top_message, message_middle = "Epochs",
                      message_bottom = "Steps", last_log = last_log, write_interval = log_write_interval)
            last_log_loss=write_log_performance_py(log_file=log_file_loss, history=history_loss.numpy().tolist(), last_log = last_log_loss, write_interval = log_write_interval)
     
      acc_test=n_matches_test/n_total_test
      bacc_test=torch.sum(torch.diagonal(confusion_matrix_test)/torch.sum(confusion_matrix_test,dim=1))/n_classes
      avg_iota_test=torch.diagonal(confusion_matrix_test)/(torch.sum(confusion_matrix_test,dim=0)+torch.sum(confusion_matrix_test,dim=1)-torch.diagonal(confusion_matrix_test))
      avg_iota_test=torch.sum(avg_iota_test)/n_classes
    
    
    #Record History
    if not (test_data is None):
      history_loss[0,epoch]=train_loss/len(trainloader)
      history_loss[1,epoch]=val_loss/len(valloader)
      history_loss[2,epoch]=test_loss/len(testloader)
      
      history_acc[0,epoch]=acc_train
      history_acc[1,epoch]=acc_val
      history_acc[2,epoch]=acc_test
      
      history_bacc[0,epoch]=bacc_train
      history_bacc[1,epoch]=bacc_val
      history_bacc[2,epoch]=bacc_test
      
      history_avg_iota[0,epoch]=avg_iota_train
      history_avg_iota[1,epoch]=avg_iota_val
      history_avg_iota[2,epoch]=avg_iota_test
    else:
      history_loss[0,epoch]=train_loss/len(trainloader)
      history_loss[1,epoch]=val_loss/len(valloader)
      
      history_acc[0,epoch]=acc_train
      history_acc[1,epoch]=acc_val
      
      history_bacc[0,epoch]=bacc_train
      history_bacc[1,epoch]=bacc_val
      
      history_avg_iota[0,epoch]=avg_iota_train
      history_avg_iota[1,epoch]=avg_iota_val

    #Trace---------------------------------------------------------------------
    if trace>=1:
      if test_data is None:
        print("Epoch: {}/{} Train Loss: {:.4f} ACC {:.4f} BACC {:.4f} AI {:.4f} | Val Loss: {:.4f} ACC: {:.4f} BACC: {:.4f}  AI: {:.4f}".format(
          epoch+1,
          epochs,
          train_loss/len(trainloader),
          acc_train,
          bacc_train,
          avg_iota_train,
          val_loss/len(valloader),
          acc_val,
          bacc_val,
          avg_iota_val))
      else:
        print("Epoch: {}/{} Train Loss: {:.4f} ACC {:.4f} BACC {:.4f} AI {:.4f} | Val Loss: {:.4f} ACC: {:.4f} BACC: {:.4f} AI {:.4f} | Test Loss: {:.4f} ACC: {:.4f} BACC: {:.4f} AI: {:.4f}".format(
          epoch+1,
          epochs,
          train_loss/len(trainloader),
          acc_train,
          bacc_train,
          avg_iota_train,
          val_loss/len(valloader),
          acc_val,
          bacc_val,
          avg_iota_val,
          test_loss/len(testloader),
          acc_test,
          bacc_test,
          avg_iota_test))
          
    #Callback-------------------------------------------------------------------
    if use_callback==True:
      if avg_iota_val>best_val_avg_iota:
        if trace>=1:
          print("Val Avg. Iota increased from {:.4f} to {:.4f}".format(best_val_avg_iota,avg_iota_val))
          print("Save checkpoint to {}".format(filepath))
        torch.save(model.state_dict(),filepath)
        best_bacc=bacc_val
        best_val_avg_iota=avg_iota_val
        best_acc=acc_val
        best_val_loss=val_loss/len(valloader)
      
      if avg_iota_val==best_val_avg_iota and acc_val>best_acc:
        if trace>=1:
          print("Val Accuracy increased from {:.4f} to {:.4f}".format(best_acc,acc_val))
          print("Save checkpoint to {}".format(filepath))
        torch.save(model.state_dict(),filepath)
        best_bacc=bacc_val
        best_acc=acc_val
        best_val_avg_iota=avg_iota_val
        best_val_loss=val_loss/len(valloader)
        
      if avg_iota_val==best_val_avg_iota and acc_val==best_acc and val_loss/len(valloader)<best_val_loss:
        if trace>=1:
          print("Val Loss decreased from {:.4f} to {:.4f}".format(best_val_loss,val_loss/len(valloader)))
          print("Save checkpoint to {}".format(filepath))
        torch.save(model.state_dict(),filepath)
        best_bacc=bacc_val
        best_acc=acc_val
        best_val_avg_iota=avg_iota_val
        best_val_loss=val_loss/len(valloader)
  
    #Check if there are furhter information for training-----------------------
    # If there are no addtiononal information. Stop training and continue
    if train_loss/len(trainloader)<1e-6 and acc_train==1 and bacc_train==1 and avg_iota_train==1:
      break
  
  #Finalize--------------------------------------------------------------------
  if use_callback==True:
    if trace>=1:
      print("Load Best Weights from {}".format(filepath))
    model.load_state_dict(torch.load(filepath,weights_only=True))


  history={
    "loss":history_loss.numpy(),
    "accuracy":history_acc.numpy(),
    "balanced_accuracy":history_bacc.numpy(),
    "avg_iota":history_avg_iota.numpy()} 
  return history

def TeClassifierTrainPrototype(model,loss_pt_fct_name , optimizer_method, scheduler_type, lr_rate,lr_min, lr_warm_up_ratio, epochs, trace,Ns,Nq,
loss_alpha, loss_margin, train_data,val_data,filepath,use_callback,n_classes,sampling_separate,sampling_shuffle,test_data=None,
log_dir=None, log_write_interval=10, log_top_value=0, log_top_total=1, log_top_message="NA"):
  
  device=('cuda' if torch.cuda.is_available() else 'cpu')
  
  if device=="cpu":
    dtype=torch.float64
    model.to(device,dtype=dtype)
  else:
    dtype=torch.double
    model.to(device,dtype=dtype)
    
  if loss_pt_fct_name =="MultiWayContrastiveLoss":
    loss_fct=multi_way_contrastive_loss(
      alpha=loss_alpha,
      margin=loss_margin)
    
  #Tensor for Saving Training History
  if not (test_data is None):
    history_loss=torch.ones(size=(3,epochs),requires_grad=False)*-100
    history_acc=torch.ones(size=(3,epochs),requires_grad=False)*-100
    history_bacc=torch.ones(size=(3,epochs),requires_grad=False)*-100
    history_avg_iota=torch.ones(size=(3,epochs),requires_grad=False)*-100
  else:
    history_loss=torch.ones(size=(2,epochs),requires_grad=False)*-100
    history_acc=torch.ones(size=(2,epochs),requires_grad=False)*-100
    history_bacc=torch.ones(size=(2,epochs),requires_grad=False)*-100
    history_avg_iota=torch.ones(size=(2,epochs),requires_grad=False)*-100
  
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
    batch_sampler=ProtoNetSampler_Train)
  
  valloader=torch.utils.data.DataLoader(
    val_data,
    batch_size=Ns+Nq,
    shuffle=False)
    
  if not (test_data is None):
    testloader=torch.utils.data.DataLoader(
      test_data,
      batch_size=Ns+Nq,
      shuffle=False)
      
  #loader_for_trained_prototpyes=torch.utils.data.DataLoader(
  #  train_data,
  #  batch_size=Ns+Nq,
  #  shuffle=False) 
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
  
  #Log file
  if not (log_dir is None):
    log_file=log_dir+"/aifeducation_state.log"
    log_file_loss=log_dir+"/aifeducation_loss.log"
    last_log=None
    last_log_loss=None
    current_step=0
    total_epochs=epochs
    
    total_steps=len(trainloader)+len(valloader)
    if not (test_data is None):
      total_steps=total_steps+len(testloader)


  for epoch in range(epochs):
    #logging
    current_step=0

    #Training------------------------------------------------------------------
    train_loss=0.0
    n_matches_train=0
    n_total_train=0
    confusion_matrix_train=torch.zeros(size=(n_classes,n_classes))
    confusion_matrix_train=confusion_matrix_train.to(device,dtype=torch.double)
    
    n_batches=0
    
    model.train(True)
    for batch in trainloader:
      model.train()
      n_batches=n_batches+1
      #assign colums of the batch
      inputs=batch["input"]
      labels=batch["labels"]

      sample_inputs=inputs[0:(n_classes*Ns)].clone()
      query_inputs=inputs[(n_classes*Ns):(n_classes*(Ns+Nq))].clone()
      
      sample_classes=labels[0:(n_classes*Ns)].clone()
      query_classes=labels[(n_classes*Ns):(n_classes*(Ns+Nq))].clone()

      sample_inputs = sample_inputs.to(device,dtype=dtype)
      query_inputs = query_inputs.to(device,dtype=dtype)
  
      sample_classes = sample_classes.to(device,dtype=dtype)
      query_classes = query_classes.to(device,dtype=dtype)

      optimizer.zero_grad()
      outputs=model(input_q=query_inputs,
      classes_q=query_classes,
      input_s=sample_inputs,
      classes_s=sample_classes,
      prediction_mode=False)
      
      loss=loss_fct(classes_q=outputs[2],distance_matrix=outputs[1],metric_scale_factor=model.get_metric_scale_factor().detach())
      loss.backward()
      optimizer.step()
      
      train_loss +=loss.item()
      model.eval()
      
      #Calc Accuracy
      pred_idx=outputs[0].max(dim=1).indices.to(dtype=torch.long,device=device)
      label_idx=query_classes.to(dtype=torch.long,device=device)
      
      match=(pred_idx==label_idx)
      n_matches_train+=match.sum().item()
      n_total_train+=outputs[0].size(0)
     

      #Calc Balanced Accuracy
      confusion_matrix_train+=multiclass_confusion_matrix(input=pred_idx,target=label_idx,num_classes=n_classes)
      
      #Update log file
      if not (log_dir is None):
        current_step+=1
        last_log=write_log_py(log_file=log_file, value_top = log_top_value, value_middle = epoch+1, value_bottom = current_step,
                  total_top = log_top_total, total_middle = epochs, total_bottom = total_steps, message_top = log_top_message, message_middle = "Epochs",
                  message_bottom = "Steps", last_log = last_log, write_interval = log_write_interval)
        last_log_loss=write_log_performance_py(log_file=log_file_loss, history=history_loss.numpy().tolist(), last_log = last_log_loss, write_interval = log_write_interval)
    
    acc_train=n_matches_train/n_total_train
    bacc_train=torch.sum(torch.diagonal(confusion_matrix_train)/torch.sum(confusion_matrix_train,dim=1))/n_classes
    avg_iota_train=torch.diagonal(confusion_matrix_train)/(torch.sum(confusion_matrix_train,dim=0)+torch.sum(confusion_matrix_train,dim=1)-torch.diagonal(confusion_matrix_train))
    avg_iota_train=torch.sum(avg_iota_train)/n_classes
    
    #Update learning rate
    if scheduler!=None:
      scheduler.step()
    
    #Calculate trained prototypes----------------------------------------------
    model.eval()
    
    class_mean_prototypes,class_label=calc_trained_prototypes_batch(
      n_classes=n_classes,
      model=model,
      data_loader=trainloader,
      device=device,
      dtype=dtype
      )
    
    model.set_trained_prototypes(
      prototypes=class_mean_prototypes,
      class_lables=class_label
      )

    #Validation----------------------------------------------------------------
    val_loss=0.0
    n_matches_val=0
    n_total_val=0
    
    confusion_matrix_val=torch.zeros(size=(n_classes,n_classes))
    confusion_matrix_val=confusion_matrix_val.to(device,dtype=torch.double)

    model.eval()
    with torch.no_grad():
      for batch in valloader:
        inputs=batch["input"]
        labels=batch["labels"]
        
        inputs = inputs.to(device,dtype=dtype)
        labels=labels.to(device,dtype=dtype)
        outputs=model(input_q=inputs,classes_q=labels,prediction_mode=False)

        loss=loss_fct(classes_q=outputs[2],distance_matrix=outputs[1],metric_scale_factor=model.get_metric_scale_factor().detach())
        val_loss +=loss.item()
        
        #Calc Accuracy
        pred_idx=outputs[0].max(dim=1).indices.to(dtype=torch.long,device=device)
        label_idx=outputs[2].to(dtype=torch.long,device=device)
        
        match=(pred_idx==label_idx)
        n_matches_val+=match.sum().item()
        n_total_val+=outputs[0].size(0)
        
        #Calc Balanced Accuracy
        confusion_matrix_val+=multiclass_confusion_matrix(input=pred_idx,target=label_idx,num_classes=n_classes)
        
        #Update log file
        if not (log_dir is None):
          current_step+=1
          last_log=write_log_py(log_file=log_file, value_top = log_top_value, value_middle = epoch+1, value_bottom = current_step,
                    total_top = log_top_total, total_middle = epochs, total_bottom = total_steps, message_top = log_top_message, message_middle = "Epochs",
                    message_bottom = "Steps", last_log = last_log, write_interval = log_write_interval)
          last_log_loss=write_log_performance_py(log_file=log_file_loss, history=history_loss.numpy().tolist(), last_log = last_log_loss, write_interval = log_write_interval)

    acc_val=n_matches_val/n_total_val
    bacc_val=torch.sum(torch.diagonal(confusion_matrix_val)/torch.sum(confusion_matrix_val,dim=1))/n_classes
    avg_iota_val=torch.diagonal(confusion_matrix_val)/(torch.sum(confusion_matrix_val,dim=0)+torch.sum(confusion_matrix_val,dim=1)-torch.diagonal(confusion_matrix_val))
    avg_iota_val=torch.sum(avg_iota_val)/n_classes
    
    #Test----------------------------------------------------------------------
    if not (test_data is None):
      test_loss=0.0
      n_matches_test=0
      n_total_test=0
      
      confusion_matrix_test=torch.zeros(size=(n_classes,n_classes))
      confusion_matrix_test=confusion_matrix_test.to(device,dtype=torch.double)
  
      model.eval()
      with torch.no_grad():
        for batch in testloader:
          inputs=batch["input"]
          labels=batch["labels"]
          
          inputs = inputs.to(device,dtype=dtype)
          labels=labels.to(device,dtype=dtype)
        
          outputs=model(inputs,prediction_mode=False)
 
          loss=loss_fct(classes_q=labels,distance_matrix=outputs[1],metric_scale_factor=model.get_metric_scale_factor().detach())
          test_loss +=loss.item()
        
          #Calc Accuracy
          pred_idx=outputs[0].max(dim=1).indices.to(dtype=torch.long,device=device)
          label_idx=labels.to(dtype=torch.long,device=device)
            
          match=(pred_idx==label_idx)
          n_matches_test+=match.sum().item()
          n_total_test+=outputs[0].size(0)
          
          #Calc Balanced Accuracy
          confusion_matrix_test+=multiclass_confusion_matrix(input=pred_idx,target=label_idx,num_classes=n_classes)
          
          #Update log file
          if not (log_dir is None):
            current_step+=1
            last_log=write_log_py(log_file=log_file, value_top = log_top_value, value_middle = epoch+1, value_bottom = current_step,
                      total_top = log_top_total, total_middle = epochs, total_bottom = total_steps, message_top = log_top_message, message_middle = "Epochs",
                      message_bottom = "Steps", last_log = last_log, write_interval = log_write_interval)
            last_log_loss=write_log_performance_py(log_file=log_file_loss, history=history_loss.numpy().tolist(), last_log = last_log_loss, write_interval = log_write_interval)
    
      
      acc_test=n_matches_test/n_total_test
      bacc_test=torch.sum(torch.diagonal(confusion_matrix_test)/torch.sum(confusion_matrix_test,dim=1))/n_classes
      avg_iota_test=torch.diagonal(confusion_matrix_test)/(torch.sum(confusion_matrix_test,dim=0)+torch.sum(confusion_matrix_test,dim=1)-torch.diagonal(confusion_matrix_test))
      avg_iota_test=torch.sum(avg_iota_test)/n_classes    
    
    #Record History
    if not (test_data is None):
      history_loss[0,epoch]=train_loss/len(trainloader)
      history_loss[1,epoch]=val_loss/len(valloader)
      history_loss[2,epoch]=test_loss/len(testloader)
      
      history_acc[0,epoch]=acc_train
      history_acc[1,epoch]=acc_val
      history_acc[2,epoch]=acc_test
      
      history_bacc[0,epoch]=bacc_train
      history_bacc[1,epoch]=bacc_val
      history_bacc[2,epoch]=bacc_test
      
      history_avg_iota[0,epoch]=avg_iota_train
      history_avg_iota[1,epoch]=avg_iota_val
      history_avg_iota[2,epoch]=avg_iota_test
    else:
      history_loss[0,epoch]=train_loss/len(trainloader)
      history_loss[1,epoch]=val_loss/len(valloader)
      
      history_acc[0,epoch]=acc_train
      history_acc[1,epoch]=acc_val
      
      history_bacc[0,epoch]=bacc_train
      history_bacc[1,epoch]=bacc_val
      
      history_avg_iota[0,epoch]=avg_iota_train
      history_avg_iota[1,epoch]=avg_iota_val

    #Trace---------------------------------------------------------------------
    if trace>=1:
      if test_data is None:
        print("Epoch: {}/{} Train Loss: {:.4f} ACC {:.4f} BACC {:.4f} AI {:.4f} | Val Loss: {:.4f} ACC: {:.4f} BACC: {:.4f}  AI: {:.4f}".format(
          epoch+1,
          epochs,
          train_loss/len(trainloader),
          acc_train,
          bacc_train,
          avg_iota_train,
          val_loss/len(valloader),
          acc_val,
          bacc_val,
          avg_iota_val))
      else:
        print("Epoch: {}/{} Train Loss: {:.4f} ACC {:.4f} BACC {:.4f} AI {:.4f} | Val Loss: {:.4f} ACC: {:.4f} BACC: {:.4f} AI {:.4f} | Test Loss: {:.4f} ACC: {:.4f} BACC: {:.4f} AI: {:.4f}".format(
          epoch+1,
          epochs,
          train_loss/len(trainloader),
          acc_train,
          bacc_train,
          avg_iota_train,
          val_loss/len(valloader),
          acc_val,
          bacc_val,
          avg_iota_val,
          test_loss/len(testloader),
          acc_test,
          bacc_test,
          avg_iota_test))
          
    #Callback-------------------------------------------------------------------
    if use_callback==True:
      if avg_iota_val>best_val_avg_iota:
        if trace>=1:
          print("Val Avg. Iota increased from {:.4f} to {:.4f}".format(best_val_avg_iota,avg_iota_val))
          print("Save checkpoint to {}".format(filepath))
        torch.save(model.state_dict(),filepath)
        best_bacc=bacc_val
        best_val_avg_iota=avg_iota_val
        best_acc=acc_val
        best_val_loss=val_loss/len(valloader)
      
      if avg_iota_val==best_val_avg_iota and acc_val>best_acc:
        if trace>=1:
          print("Val Accuracy increased from {:.4f} to {:.4f}".format(best_acc,acc_val))
          print("Save checkpoint to {}".format(filepath))
        torch.save(model.state_dict(),filepath)
        best_bacc=bacc_val
        best_acc=acc_val
        best_val_avg_iota=avg_iota_val
        best_val_loss=val_loss/len(valloader)
        
      if avg_iota_val==best_val_avg_iota and acc_val==best_acc and val_loss/len(valloader)<best_val_loss:
        if trace>=1:
          print("Val Loss decreased from {:.4f} to {:.4f}".format(best_val_loss,val_loss/len(valloader)))
          print("Save checkpoint to {}".format(filepath))
        torch.save(model.state_dict(),filepath)
        best_bacc=bacc_val
        best_acc=acc_val
        best_val_avg_iota=avg_iota_val
        best_val_loss=val_loss/len(valloader)
          
    #Check if there are furhter information for training-----------------------
    # If there are no addtiononal information. Stop training and continue
    if train_loss/len(trainloader)<1e-6 and acc_train==1 and bacc_train==1 and avg_iota_train==1:
      break
  
  #Finalize--------------------------------------------------------------------
  if use_callback==True:
    if trace>=1:
      print("Load Best Weights from {}".format(filepath))
    model.load_state_dict(torch.load(filepath,weights_only=True))
    #safetensors.torch.load_model(model=model,filename=filepath)


  history={
    "loss":history_loss.numpy(),
    "accuracy":history_acc.numpy(),
    "balanced_accuracy":history_bacc.numpy(),
    "avg_iota":history_avg_iota.numpy()} 

  return history


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
