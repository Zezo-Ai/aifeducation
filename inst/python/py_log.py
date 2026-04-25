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

import csv
import time

def _write_dict(file,
                vt, tt, mt,
                vm, tm, mm,
                vb, tb, mb):
  f = open(file, "w", newline = "")
  fieldnames = ["value", "total", "message"]
  writer = csv.DictWriter(f, fieldnames = fieldnames, dialect = 'unix')
  writer.writeheader()
  row = lambda v, t, m : { 'value': v, 'total': t, 'message': m }
  writer.writerow(row(v = vt, t = tt, m = mt))
  writer.writerow(row(v = vm, t = tm, m = mm))
  writer.writerow(row(v = vb, t = tb, m = mb))
  f.close()

def _write_history(file, history):
  f = open(file, "w", newline = "")
  writer = csv.writer(f, dialect = 'unix')
  writer.writerows(history)
  f.close()
  
def _write(write_fn, args_fn,
           last_log, write_interval):

  if args_fn["file"] == None:
    return None
  
  log_time = None
  diff = float("inf") if last_log == None else time.time() - last_log
        
  if diff > write_interval:
    try:
      write_fn(**args_fn)
      log_time = time.time()
    except:
      log_time = last_log

  return log_time

def write_log_py(log_file, 
                 value_top = 0, total_top = 1, message_top = "NA",
                 value_middle = 0, total_middle = 1, message_middle = "NA",
                 value_bottom = 0, total_bottom = 1, message_bottom = "NA",
                 last_log = None, write_interval = 2):
                   
  args_fn = { "file": log_file,
              "vt": value_top, "tt": total_top, "mt": message_top, 
              "vm": value_middle, "tm": total_middle, "mm": message_middle,
              "vb": value_bottom, "tb": total_bottom, "mb": message_bottom }
              
  return _write(_write_dict, args_fn, last_log, write_interval)

def write_log_performance_py(log_file, history, 
                             last_log = None, write_interval = 2):
                               
  args_fn = { "file": log_file, "history": history }
              
  return _write(_write_history, args_fn, last_log, write_interval)


class LogWriter:
  def __init__(self,log_file,log_file_loss ,value_top = 1, value_middle = 1, value_bottom = 1,
                  total_top = 2, total_middle = 2, total_bottom = 2, message_top = "Top", message_middle = "Middle",
                  message_bottom = "Bottom", last_log = None, write_interval = 2):
    self.log_file=log_file
    self.log_file_loss=log_file_loss
    self.value_top=value_top
    self.value_middle=value_middle
    self.value_bottom=value_bottom
    self.total_top=total_top
    self.total_middle=total_middle
    self.total_bottom=total_bottom
    self.message_top=message_top
    self.message_middle=message_middle
    self.message_bottom=message_bottom
    self.last_log=last_log
    self.last_log_loss=last_log
    self.write_interval=write_interval
  def set_value(self,value,level):
    if level=="top":
      self.value_top=value
    elif level=="middle":
      self.value_middle=value
    elif level=="bottom":
      self.value_bottom=value
  def inc_value(self,level):
    if level=="top":
      self.value_top+=1
    elif level=="middle":
      self.value_middle+=1
    elif level=="bottom":
      self.value_bottom+=1
  def reset_value(self,level):
    if level=="top":
      self.value_top=0
    elif level=="middle":
      self.value_middle=0
    elif level=="bottom":
      self.value_bottom=0   
  def set_history_loss(self,history_loss):
    self.history_loss=history_loss
  def write_log(self):
    if not (self.log_file is None):
      self.last_log=write_log_py(log_file=self.log_file, value_top = self.value_top, value_middle = self.value_middle, value_bottom = self.value_bottom,
                    total_top = self.total_top, total_middle = self.total_middle, total_bottom = self.total_bottom, message_top = self.message_top, message_middle = self.message_middle,
                    message_bottom = self.message_bottom, last_log = self.last_log, write_interval = self.write_interval)
  def write_history_log(self,history_loss):
    if not (self.log_file is None):
      self.last_log_loss=write_log_performance_py(log_file=self.log_file_loss, history=history_loss.tolist(), last_log = self.last_log_loss, write_interval = self.write_interval)
