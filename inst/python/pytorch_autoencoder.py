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

def calc_SquaredCovSum(x):
  times=x.size(dim=1)
  cov_sum=0.0

  for i in range(times):
    current_time_point=torch.squeeze(x[:,i,:])
    if current_time_point.dim()>1:
      current_cases_index=torch.nonzero(torch.sum(current_time_point,axis=1))
      if current_cases_index.size(dim=0)>1:
        current_cases=torch.squeeze(current_time_point[current_cases_index])
        covariance=torch.cov(torch.transpose(current_cases,dim0=0,dim1=1))
        covariance=torch.square(covariance)
        cov_sum=cov_sum+(torch.sum(covariance)-torch.sum(torch.diag(covariance)))/current_cases.size(dim=0)
  cov_sum=cov_sum/times
  return cov_sum

class LSTMAutoencoder_with_Mask_PT(torch.nn.Module):
    def __init__(self,times, features_in,features_out,noise_factor,pad_value):
      super().__init__()
      self.features_in=features_in
      self.features_out=features_out
      self.sequence_length=times
      self.noise_factor=noise_factor
      self.difference=self.features_in-self.features_out
      self.PackAndMasking_PT=PackAndMasking_PT()
      self.UnPackAndMasking_PT=UnPackAndMasking_PT(sequence_length=self.sequence_length)
      
      self.encoder_1=torch.nn.LSTM(
        input_size=self.features_in,
        hidden_size=math.ceil(self.features_in-self.difference*(1/2)),
        batch_first=True,
        bias=True)
        
      self.latent_space=torch.nn.LSTM(
        input_size=math.ceil(self.features_in-self.difference*(1/2)),
        hidden_size=self.features_out,
        batch_first=True,
        bias=True)
        
      self.decoder_1=torch.nn.LSTM(
        input_size=self.features_out,
        hidden_size=math.ceil(self.features_in-self.difference*(1/2)),
        batch_first=True,
        bias=True)
      
      self.output=torch.nn.LSTM(
        input_size=math.ceil(self.features_in-self.difference*(1/2)),
        hidden_size=self.features_in,
        batch_first=True,
        bias=True)
        
      if not pad_value==0:
        self.switch_pad_value_start=layer_switch_pad_values(pad_value_old=pad_value,pad_value_new=0)
        self.switch_pad_value_final=layer_switch_pad_values(pad_value_old=0,pad_value_new=pad_value)
      else:
        self.switch_pad_value_start=None
        self.switch_pad_value_final=None
        
    def forward(self, x, encoder_mode=False, return_scs=False):
      #Swtich padding value if necessary
      if not self.switch_pad_value_start==None:
        x=self.switch_pad_value_start(x)
        
      if encoder_mode==False:
        if self.training==True:
          mask=self.get_mask(x)
          x=x+self.add_noise(x)
          x=~mask*x
        x=self.PackAndMasking_PT(x)
        x=self.encoder_1(x)[0]
        latent_space=self.latent_space(x)[0]
        x=self.decoder_1(latent_space)[0]
        x=self.output(x)[0]
        x=self.UnPackAndMasking_PT(x)
        #Switch padding value back if necessary
        if not self.switch_pad_value_start==None:
          x=self.switch_pad_value_final(x)
        if return_scs==False:
          return x
        else:
          return x, calc_SquaredCovSum(self.UnPackAndMasking_PT(latent_space))
      
      elif encoder_mode==True:
        x=self.PackAndMasking_PT(x)
        x=self.encoder_1(x)[0]
        x=self.latent_space(x)[0]
        x=self.UnPackAndMasking_PT(x)
        #Switch padding value back if necessary
      if not self.switch_pad_value_start==None:
        x=self.switch_pad_value_final(x)
      return x
    def get_mask(self,x):
      time_sums=torch.sum(x,dim=2)
      mask=(time_sums==0)
      mask_long=torch.reshape(torch.repeat_interleave(mask,repeats=self.features_in,dim=1),(x.size(dim=0),x.size(dim=1),self.features_in))
      mask_long=mask_long.to(x.device)
      return mask_long
    def add_noise(self, x):
      noise=self.noise_factor*torch.rand(size=x.size())
      noise=noise.to(x.device)
      return(noise)
      
class DenseAutoencoder_with_Mask_PT(torch.nn.Module):
    def __init__(self, features_in,features_out,noise_factor,pad_value,orthogonal_method):
      super().__init__()
      self.features_in=features_in
      self.features_out=features_out
      self.noise_factor=noise_factor
      self.difference=self.features_in-self.features_out
      
      self.param_w1=torch.nn.Parameter(torch.randn(math.ceil(self.features_in-self.difference*(2/3)),self.features_in))
      self.param_w2=torch.nn.Parameter(torch.randn(math.ceil(self.features_in-self.difference*(1/3)),math.ceil(self.features_in-self.difference*(2/3))))
      self.param_w3=torch.nn.Parameter(torch.randn(self.features_out,math.ceil(self.features_in-self.difference*(1/3))))
      
      if not orthogonal_method=="None":
        torch.nn.utils.parametrizations.orthogonal(module=self, name="param_w1",orthogonal_map=orthogonal_method)
        torch.nn.utils.parametrizations.orthogonal(module=self, name="param_w2",orthogonal_map=orthogonal_method)
        torch.nn.utils.parametrizations.orthogonal(module=self, name="param_w3",orthogonal_map=orthogonal_method)
      
      if not pad_value==0:
        self.switch_pad_value_start=layer_switch_pad_values(pad_value_old=pad_value,pad_value_new=0)
        self.switch_pad_value_final=layer_switch_pad_values(pad_value_old=0,pad_value_new=pad_value)
      else:
        self.switch_pad_value_start=None
        self.switch_pad_value_final=None

    def forward(self, x, encoder_mode=False, return_scs=False):
      #Swtich padding value if necessary
      if not self.switch_pad_value_start==None:
        x=self.switch_pad_value_start(x)
      if encoder_mode==False:
        #Add noise
        if self.training==True:
          mask=self.get_mask(x)
          x=x+self.add_noise(x)
          x=~mask*x
        
        #Encoder
        x=torch.nn.functional.tanh(torch.nn.functional.linear(x,weight=self.param_w1))
        x=torch.nn.functional.tanh(torch.nn.functional.linear(x,weight=self.param_w2))
        
        #Latent Space
        latent_space=torch.nn.functional.tanh(torch.nn.functional.linear(x,weight=self.param_w3))

        #Decoder
        x=torch.nn.functional.tanh(torch.nn.functional.linear(latent_space,weight=torch.transpose(self.param_w3,dim0=1,dim1=0)))
        x=torch.nn.functional.tanh(torch.nn.functional.linear(x,weight=torch.transpose(self.param_w2,dim0=1,dim1=0)))
        x=torch.nn.functional.tanh(torch.nn.functional.linear(x,weight=torch.transpose(self.param_w1,dim0=1,dim1=0)))

        
        #Switch padding value back if necessary
        if not self.switch_pad_value_start==None:
          x=self.switch_pad_value_final(x)

        if return_scs==False:
          return x
        else:
          return x, calc_SquaredCovSum(latent_space)
      elif encoder_mode==True:
        #Encoder
        x=torch.nn.functional.tanh(torch.nn.functional.linear(x,weight=self.param_w1))
        x=torch.nn.functional.tanh(torch.nn.functional.linear(x,weight=self.param_w2))
        
        #Latent Space
        x=torch.nn.functional.tanh(torch.nn.functional.linear(x,weight=self.param_w3))
        #Switch padding value back if necessary
        if not self.switch_pad_value_start==None:
          x=self.switch_pad_value_final(x)
        return x
      
    def get_mask(self,x):
      time_sums=torch.sum(x,dim=2)
      mask=(time_sums==0)
      mask_long=torch.reshape(torch.repeat_interleave(mask,repeats=self.features_in,dim=1),(x.size(dim=0),x.size(dim=1),self.features_in))
      mask_long=mask_long.to(x.device)
      return mask_long
    def add_noise(self, x):
      noise=self.noise_factor*torch.rand(size=x.size())
      noise=noise.to(x.device,x.dtype)
      return(noise)
    
class ConvAutoencoder_with_Mask_PT(torch.nn.Module):
    def __init__(self, features_in,features_out,noise_factor):
      super().__init__()
      self.features_in=features_in
      self.features_out=features_out
      self.noise_factor=noise_factor
      self.difference=self.features_in-self.features_out
      self.stride=1
      self.kernel_size=2
      
      #dilation of 1 means no dilation
      self.dilation=1
      
      self.param_w1=torch.nn.Parameter(torch.randn(math.ceil(self.features_in-self.difference*(1/2)),self.features_in,self.kernel_size))
      self.param_w2=torch.nn.Parameter(torch.randn(self.features_out,math.ceil(self.features_in-self.difference*(1/2)),self.kernel_size))
      
      self.sequence_reduction=torch.nn.AvgPool1d(kernel_size=(self.kernel_size),stride=1,padding=0)
      if not orthogonal_method=="None":
        torch.nn.utils.parametrizations.orthogonal(self, "param_w1",orthogonal_map="householder")
        torch.nn.utils.parametrizations.orthogonal(self, "param_w2",orthogonal_map="householder")

    def forward(self, x, encoder_mode=False, return_scs=False):
      if encoder_mode==False:
        #Add noise
        if self.training==True:
          mask=self.get_mask(x)
          x=x+self.add_noise(x)
          x=~mask*x
        
        #Change position of time and features
        x=torch.transpose(x, dim0=1, dim1=2)
        
        #Encoder
        x=torch.nn.functional.tanh(torch.nn.functional.conv1d(x,weight=self.param_w1,stride=self.stride,padding='same',dilation=self.dilation))

        #Latent Space
        latent_space=torch.nn.functional.tanh(torch.nn.functional.conv1d(x,weight=self.param_w2,stride=self.stride,padding='same',dilation=self.dilation))
        latent_space=torch.transpose(latent_space, dim0=1, dim1=2)
        latent_space=~self.get_mask(latent_space)*latent_space
        latent_space=torch.transpose(latent_space, dim0=1, dim1=2)

        #Decoder
        x=torch.nn.functional.tanh(torch.nn.functional.conv_transpose1d(latent_space,weight=self.param_w2,stride=self.stride,padding=0,output_padding=0,dilation=self.dilation))
        x=self.sequence_reduction(x)
        x=torch.nn.functional.tanh(torch.nn.functional.conv_transpose1d(x,weight=self.param_w1,stride=self.stride,padding=0,output_padding=0,dilation=self.dilation))
        x=self.sequence_reduction(x)
        
        #Change position of time and features
        x=torch.transpose(x, dim0=1, dim1=2)
        
        if return_scs==False:
          return x
        else:
          latent_space=torch.transpose(latent_space, dim0=1, dim1=2)
          return x, calc_SquaredCovSum(latent_space)
      elif encoder_mode==True:
        #Change position of time and features
        x=torch.transpose(x, dim0=1, dim1=2)
        #Encoder
        x=torch.nn.functional.tanh(torch.nn.functional.conv1d(x,weight=self.param_w1,stride=self.stride,padding='same'))

        #Latent Space
        x=torch.nn.functional.tanh(torch.nn.functional.conv1d(x,weight=self.param_w2,stride=self.stride,padding='same'))
        #Change position of time and features
        x=torch.transpose(x, dim0=1, dim1=2)
        x=~self.get_mask(x)*x
        return x
      
    def get_mask(self,x):
      device=('cuda' if torch.cuda.is_available() else 'cpu')
      time_sums=torch.sum(x,dim=2)
      mask=(time_sums==0)
      mask_long=torch.reshape(torch.repeat_interleave(mask,repeats=x.size(dim=2),dim=1),(x.size(dim=0),x.size(dim=1),x.size(dim=2)))
      mask_long=mask_long.to(device)
      return mask_long
    def add_noise(self, x):
      device=('cuda' if torch.cuda.is_available() else 'cpu')
      noise=self.noise_factor*torch.rand(size=x.size())
      noise=noise.to(device)
      return(noise)


def run_epoch_autoencoder(model,dataloader,loss_fct,optimizer,scaler,scheduler,amp,epoch,device,current_dtype,cblock,metric_storage,logger):
  total_loss=0.0
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
      if cblock=="train":
        optimizer.zero_grad()
      with torch.autocast(device_type=device, dtype=None, enabled=amp):  
        outputs=model(inputs,encoder_mode=False)
        loss=loss_fct(outputs,labels)
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
    #Update log file
    logger.inc_value("bottom")
    logger.write_log()
    logger.write_history_log(metric_storage["loss"])
  #Calc final metrics for epoch
  results={"loss":total_loss/len(dataloader)}
  #Save metrics
  add_metrics(
    metrics=results,
    storage=metric_storage,
    cblock=cblock,
    epoch=epoch
  )
  return results

def check_and_set_checkpoints_loss(use_callback,model,filepath,epoch,metric_storage,best_val_loss,val_loss):
  if use_callback==True:
      if val_loss<=best_val_loss:
        torch.save(model.state_dict(),filepath)
        best_val_loss=val_loss
        metric_storage["checkpoints"][epoch]=1
  return best_val_loss
    
def AutoencoderTrain_PT_with_Datasets(model,optimizer_method,scheduler_type,amp, lr_rate,lr_min, lr_warm_up_ratio, epochs, trace,batch_size,
train_data,val_data,filepath,use_callback,
log_dir=None, log_write_interval=10, log_top_value=0, log_top_total=1, log_top_message="NA"):
  #Set test data to None
  test_data=None
  #Prepare model
  device=get_device()
  current_dtype=get_dtype(device)
  model.to(device=device,dtype=current_dtype)
  #Prepare loss function
  loss_fct=torch.nn.MSELoss()
  loss_fct.to(device,dtype=current_dtype)
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
  #Tensor for Saving Training History
    #Numpys for Saving Training History
  metric_storage=create_metric_storage(
    metric_names=["loss"],
    epochs=epochs,
    inc_test=True if not (test_data is None) else False
  )
  # Init checkpoint values
  best_val_loss=float('inf')
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

  for epoch in range(epochs):
    train_results=run_epoch_autoencoder(
      model=model,
      dataloader=trainloader,
      optimizer=optimizer,
      scaler=amp_scaler,
      amp=amp,
      scheduler=scheduler,
      loss_fct=loss_fct,
      epoch=epoch,
      device=device,
      current_dtype=current_dtype,
      cblock="train",
      metric_storage=metric_storage,
      logger=logger
    )
    val_results=run_epoch_autoencoder(
      model=model,
      dataloader=valloader,
      loss_fct=loss_fct,
      optimizer=optimizer,
      scaler=amp_scaler,
      amp=amp,
      scheduler=scheduler,
      epoch=epoch,
      device=device,
      current_dtype=current_dtype,
      cblock="val",
      metric_storage=metric_storage,
      logger=logger
    )
    if testloader is not None:
      test_results=run_epoch_autoencoder(
        model=model,
        dataloader=testloader,
        optimizer=optimizer,
        scaler=amp_scaler,
        amp=amp,
        scheduler=scheduler,
        loss_fct=loss_fct,
        epoch=epoch,
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
    best_val_loss=check_and_set_checkpoints_loss(
      use_callback=use_callback,
      model=model,
      filepath=filepath,
      epoch=epoch,
      metric_storage=metric_storage,
      best_val_loss=best_val_loss,
      val_loss=val_results["loss"]
    )
    #Trace---------------------------------------------------------------------
    print_epoch_results(
      trace=trace,
      loss_only=True,
      metric_storage=metric_storage,
      epoch=epoch,
      epochs=epochs,
      metric_criterion="loss",
      best_metric=None,
      best_loss=best_val_loss
    )
  #Finalize--------------------------------------------------------------------
  if use_callback==True:
    if trace>=1:
      print("Load Best Weights from {}".format(filepath))
    model.load_state_dict(torch.load(filepath,weights_only=True))

  return metric_storage

@torch.inference_mode()
def TeFeatureExtractorBatchExtract(model,dataset,batch_size):
  
  device=('cuda' if torch.cuda.is_available() else 'cpu')
  
  if device=="cpu":
    dtype=torch.float
    model.to(device,dtype=dtype)
  else:
    dtype=torch.float
    model.to(device,dtype=dtype)
    
  model.eval()
  predictionloader=torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False)

  iteration=0
  for batch in predictionloader:
    inputs=batch["input"]
    inputs = inputs.to(device,dtype=dtype)
    predictions=model(inputs,encoder_mode=True)
    
    if iteration==0:
      predictions_list=predictions.to("cpu")
    else:
      predictions_list=torch.concatenate((predictions_list,predictions.to("cpu")), axis=0, out=None)
    iteration+=1
  
  return predictions_list
