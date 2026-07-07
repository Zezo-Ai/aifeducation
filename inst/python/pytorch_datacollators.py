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

class MetaLernerBatchSampler(torch.utils.data.sampler.Sampler):
    #Ns Number of examples per class in sample set (k-shot)
    #Nq number of examples per class in the query set (k-shot)
    #targets pytorch tensor containing the classes/categories
    def __init__(self, targets, Ns,Nq,separate,shuffle):
        # build data for sampling here
        self.Ns=Ns
        self.Nq=Nq
        self.separate=separate
        self.shuffle=shuffle

        #Get the available classes in targets
        self.classes=torch.unique(targets).numpy()
        #Get the number of classes
        self.n_classes=len(self.classes)
        #Calculate the batch size depending on Ns and Nq
        self.batch_size=self.n_classes*(self.Ns+self.Nq)
        
        #Create dictonary that contains the indexes sorted for every class
        self.indices_per_class={}
        #Create dictornary thats sorts the number of cases per class
        self.cases_per_class={}
        #Gather indicies per class and cases per class
        for c in self.classes:
          self.indices_per_class[c]=torch.where(targets==c)[0]
          self.cases_per_class[c]=len(self.indices_per_class[c])
        
         #Create dictonary that contains the indexes sorted for every class and query/sample
        self.query_indices_per_class={}
        self.sample_indices_per_class={}
        #Create dictornary that sorts the number of cases per class and query/sample
        self.query_cases_per_class={}
        self.sample_cases_per_class={}

        #Split indices per class into a sample and query sample if separate is True
        if self.separate is True:
          for c in self.classes:
            #Calculate number of cases for sample
            n_sample=int(round(self.Ns/(self.Ns+self.Nq)*self.cases_per_class[c]))
            n_sample=max(1,n_sample)
            n_sample=min(n_sample,(self.cases_per_class[c]-1))
            #Calculate number of cases for query
            n_query=self.cases_per_class[c]-n_sample
            #Create permutation in order to create a random sample
            permutation=self.indices_per_class[c][torch.randperm(self.cases_per_class[c])]
            #Assign indices
            self.sample_indices_per_class[c]=permutation[np.array(range(0,n_sample))]
            self.query_indices_per_class[c]=permutation[np.array(range(n_sample,self.cases_per_class[c]))]
            #Calculate number of cases
            self.sample_cases_per_class[c]=len(self.sample_indices_per_class[c])
            self.query_cases_per_class[c]=len(self.query_indices_per_class[c])
        else:
          #Create a random permutation if separate is False and shuffle is False
          #If shuffle is True random sampling is applied during iter
          if self.shuffle is False:
            for c in self.classes:
              self.indices_per_class[c]=self.indices_per_class[c][torch.randperm(self.cases_per_class[c])]
          
        #Calculate number of batches
        self.number_batches=self.cases_per_class[max(self.cases_per_class,key=self.cases_per_class.get)]//(self.Ns+self.Nq)
        
    def __iter__(self):
      for current_iter in range(self.number_batches):
        #Create list for saving the results per class temporarily 
        batch_sample=[]
        batch_query=[]
      
        if self.separate is False:
          if self.shuffle is True:
            for c in self.classes:
              #Calculate permutations for the random sample for each class
              permutations=self.indices_per_class[c][torch.randperm(self.cases_per_class[c])]

              #Calculat the indexes for selecting the first Ns+Nq indices
              if(self.cases_per_class[c]>=self.Ns+self.Nq):
                ids_sample=np.array(range(0,self.Ns))
                ids_query=np.array(range(self.Ns,(self.Ns+self.Nq)))
              else:
                #For the case that the number of cases is lower as Ns+Nq adjust proportional
                tmp_Ns=max(1,math.floor(self.cases_per_class[c]*self.Ns/(self.Ns+self.Nq))-1)
                tmp_Nq=self.cases_per_class[c]-tmp_Ns
                ids_sample=np.array(range(0,tmp_Ns))
                ids_query=np.array(range(tmp_Ns,(tmp_Ns+tmp_Nq)))

              #Extract the final indices
              perm_sample=permutations[ids_sample].numpy()
              perm_query=permutations[ids_query].numpy()
              
              #Add them to batch
              batch_sample.extend(perm_sample)
              batch_query.extend(perm_query)
          else:
            for c in self.classes:
              #Calculate indices. If the end of all cases is reached for this class
              #start at beginng and fill the list
              index_shift= (1+current_iter)*(self.Ns+self.Nq)
              ids_sample=np.array(range((0+index_shift),(self.Ns+index_shift)))%self.cases_per_class[c]
              ids_query=np.array(range((self.Ns+index_shift),((self.Ns+self.Nq)+index_shift)))%self.cases_per_class[c]
              
              #Extract the final indices
              perm_sample=self.indices_per_class[c][ids_sample].numpy()
              perm_query=self.indices_per_class[c][ids_query].numpy()
            
              #Add them to batch
              batch_sample.extend(perm_sample)
              batch_query.extend(perm_query)
        if self.separate is True:
          if self.shuffle is True:
            for c in self.classes:
              #Calculate permutations for the random sample for each class
              permutations_sample=self.sample_indices_per_class[c][torch.randperm(self.sample_cases_per_class[c])]
              permutations_query=self.query_indices_per_class[c][torch.randperm(self.query_cases_per_class[c])]
              
              #Calculat the indexes for selecting the first Ns+Nq indices
              ids_sample=np.array(range(0,self.Ns))
              ids_query=np.array(range(0,self.Nq))
              
              #Extract the final indices
              perm_sample=permutations_sample[ids_sample].numpy()
              perm_query=permutations_query[ids_query].numpy()
                
              #Add them to batch
              batch_sample.extend(perm_sample)
              batch_query.extend(perm_query)
          else:
            #Calculate indices. If the end of all cases is reached for this class
            #start at beginng and fill the list
            index_shift_sample= (1+current_iter)*self.Ns
            index_shift_query= (1+current_iter)*self.Nq
            for c in self.classes:
              ids_sample=np.array(range((0+index_shift_sample),(self.Ns+index_shift_sample)))%self.sample_cases_per_class[c]
              ids_query=np.array(range((0+index_shift_query),(self.Nq+index_shift_query)))%self.query_cases_per_class[c]
              
              #Extract the final indices
              perm_sample=self.sample_indices_per_class[c][ids_sample].numpy()
              perm_query=self.query_indices_per_class[c][ids_query].numpy()
              
              #Add them to batch
              batch_sample.extend(perm_sample)
              batch_query.extend(perm_query)
        #Create the final batch
        #Add first the sample of all classes and then the query of all classes
        batch=[]
        batch=batch_sample+batch_query
        yield batch
      
    def __len__(self):
      return self.number_batches
