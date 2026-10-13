from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader
import os
import torch

from torchvision.transforms import v2

from PIL import Image
import numpy as np
import re
import pandas as pd
import sys

join = os.path.join

class Fold_Dataset(Dataset):
    def __init__(self, root, meta=0, datasets=-1, miss_mod=False, data_prefix=None):
        self.files = {"Result": [], "Data": [], "Meta": []}
        self.data_prefix = data_prefix if data_prefix is not None else "/data/edu/"
        self.classification = []
        self.types = []
        self.tasks = []
        self.enc_task = []
        self.max_task = 0
        self.indices = []
        
        csv_files = ["train_final.csv", "val_final.csv", "test_final.csv"]
        if miss_mod:
            csv_files = ["train_miss.csv", "val_miss.csv", "test_miss.csv"]
            
        aux = pd.concat([pd.read_csv(join(root, f)) for f in csv_files])
        
        if miss_mod:
            aux=aux[aux["Type"]!=1]
            aux.loc[aux['Type'] > 0, 'Type'] -= 1
        n_meta = 0
        
        for col in aux.columns:
            if re.match("Meta.*",col):
                n_meta+=1
        
        for _,element in aux.iterrows():
            if datasets==-1 or element["Type"]==datasets:
                if datasets!=-1:
                    element["Type"]=0
                metadata = []
                self.files["Data"].append({})
                dimensions = [int(x) for x in str(element["Dim"])]
                self.types.append(dimensions)
                if element["Result"]+1> self.max_task:
                    self.max_task = element["Result"]+1
                self.files["Result"].append(element["Result"])
                accumulate = 0
                for n in range(0, len(dimensions)):
                    while not pd.notnull(element["Mod" + str(n+accumulate)]) :
                        accumulate+=1
                    self.files["Data"][len(self.files["Data"])-1]["Mod" + str(n)]=element["Mod" + str(n+accumulate)]

                for q in range(0,n_meta):
                    if pd.notnull(element["Meta"+str(q)]):
                        metadata.append(float(element["Meta"+str(q)]))
                if len(metadata)>n_meta:
                    metadata = metadata[0:meta]
                else:
                    dif = n_meta-len(metadata)
                    for _ in range(0,dif):
                        metadata.append(0)
                metadata.append(element["Type"])
                if element["Type"] not in self.tasks:
                    self.tasks.append(element["Type"])
                    if len(self.enc_task)<int(element["Type"]+1):
                        for _ in range(int(element["Type"])+1- len(self.enc_task)):
                            self.enc_task.append("")
                    self.enc_task[int(element["Type"])] = element["Dim"]
                    
                self.files["Meta"].append(metadata)
                
        total =0
        count=0
        aux = []
        for val in self.files["Meta"]:
            if val[-1] not in aux:
                aux.append(val[-1])
        
            
        self.dist = {key:[] for key in range(0,int(len(aux)))}
        for i,label in enumerate(self.files["Result"]):
            task_id = int(self.files["Meta"][i][-1])
            while len(self.dist[task_id])<=label:
                self.dist[task_id].append(0)
                
            self.dist[task_id][int(label)]+=1
        
        for k,arr in self.dist.items():
            for idx,v in enumerate(arr):
                count+=1
                self.dist[k][idx]=1/(v+1e-8)
                total+=self.dist[k][idx]
                
        for k,arr in self.dist.items():
            for idx,_ in enumerate(arr):
                self.dist[k][idx]/=(total/count)
            

        self.full_len = len(self.files["Result"])
        self.indices = np.arange(self.full_len) 

    def set_indices(self, indices):
        """stratified K fold select the samples"""
        self.indices = indices
        self.len = len(indices)

    def __len__(self):
        return self.len
    
    def get_dist(self):       
        return self.dist       
    
    def max_lab(self):
        return int(self.max_task)
    
    def obtain_tasks(self):
        return int(len(self.tasks))

    def task_types(self):
        return self.enc_task

    def __getitem__(self, item):
        actual_idx = self.indices[item]
        
        train_transforms = v2.RandomChoice([v2.RandomAffine(scale=(0.7, 1.5),degrees=(0,15),translate = (0.15,0.15)),
                                     v2.ElasticTransform(alpha=50),
                                     v2.GaussianBlur(kernel_size = [7,7])
                                     ])
        images = {}

        #transformed = {}
        for i in range(0,len(self.types[actual_idx])):
            actual = len(images)
            images["Mod" + str(actual)] = np.load(os.path.join(self.data_prefix, self.files["Data"][actual_idx]["Mod" + str(i)]))
            if self.files["Data"][actual_idx]["Mod" + str(i)].split(".")[-1] == "npz":
                images["Mod" + str(actual)] = images["Mod" + str(actual)][list(images["Mod" + str(actual)].keys())[0]]

            images["Mod" + str(actual)] = torch.from_numpy(images["Mod" + str(actual)])
            images["Mod" + str(actual)] = (images["Mod" + str(actual)] - torch.min(images["Mod" + str(actual)]))
            images["Mod" + str(actual)] = images["Mod" + str(actual)]/torch.max(images["Mod" + str(actual)])
            images["Mod" + str(actual)] = (images["Mod" + str(actual)]  - torch.mean(images["Mod" + str(actual)] )) / torch.std(images["Mod" + str(actual)] )
            diff = self.types[actual_idx][i]+1-len(images["Mod" + str(actual)].shape)
            if diff>0:
                for _ in range(0, diff):
                    images["Mod" + str(actual)] = images["Mod" + str(actual)].unsqueeze(dim=0)
            elif diff<0:
                for _ in range(0, -diff):
                    images["Mod" + str(actual)] = images["Mod" + str(actual)].squeeze(dim=0)
            if self.val==0:
                #transformed["Mod"+str(i)] = transforms(images["Mod"+str(i)])
                images["Mod" + str(actual)] = train_transforms(images["Mod" + str(actual)])


        y = self.files["Result"][actual_idx]
        types = self.types[actual_idx]

        
        return images, y, torch.tensor(self.files["Meta"][actual_idx]), types