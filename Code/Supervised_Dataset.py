import os
import torch
import time
from torch.utils.data import Dataset, DataLoader
import torchvision

from torchvision.transforms import v2

import torchio.transforms as torcht

from PIL import Image
import numpy as np
import re
import pandas as pd
import nibabel as nib
import glob
import sys
import skimage.transform as skTrans
import torchio as tio
import matplotlib.pyplot as plt

join = os.path.join

class Supervised_Dataset(Dataset):
    def __init__(self,root,val=0, meta = 0, datasets = -1,  miss_mod = False, data_prefix=None):
        self.files = {}
        self.val = val
        self.data_prefix = data_prefix if data_prefix is not None else "/data/edu/"
        self.classification = []
        self.types = []
        self.tasks = []
        self.enc_task = []
        self.max_task = 0
        self.files["Result"] = []
        self.files["Data"] = []
        self.files["Meta"] = []
        self.dist = {}
        
        self.transform = tio.Compose([
            tio.RandomAffine(scales=(0.7, 1.5), degrees=15, translation=0.15),
            tio.RandomElasticDeformation(num_control_points=5, max_displacement=7),
            tio.RandomBlur(std=(0, 1)),
        ])

        if val==0:
            aux = join(root, r"train_final.csv") if not miss_mod else join(root, r"train_miss.csv")
        elif val ==1:
            aux = join(root, r"val_final.csv") if not miss_mod else join(root, r"val_miss.csv")
        else:
            aux = join(root, r"test_final.csv") if not miss_mod else join(root, r"test_miss.csv")
        aux = pd.read_csv(aux)
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
        self.dist = {key:[] for key in range(0,int(self.max_task)-1)}
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
            

        self.len = len(self.files["Result"])

    def get_proportion(self):
        num_positives = np.sum(self.files["Result"])
        num_negatives = self.len - num_positives
        pos_weight = num_negatives / (num_positives+num_negatives)

        return pos_weight
    
    def get_dist(self):       
        return self.dist
            
    
    def max_lab(self):
        return int(self.max_task)
    
    def obtain_tasks(self):
        return int(len(self.tasks))

    def task_types(self):
        return self.enc_task

    def __getitem__(self, item):
        images = {}
        for i in range(len(self.types[item])):
            path = os.path.join(self.data_prefix, self.files["Data"][item]["Mod" + str(i)])
            
            # Faster loading: use mmap_mode if files are large
            img_np = np.load(path)
            if path.endswith(".npz"):
                img_np = img_np[img_np.files[0]]
            
            img_tensor = torch.from_numpy(img_np).float()

            # Vectorized Normalization (Z-score)
            # Often, just Z-score is enough; min-max followed by Z-score is redundant
            mean, std = img_tensor.mean(), img_tensor.std()
            img_tensor = (img_tensor - mean) / (std + 1e-8)

            # Fast dimension adjustment
            target_dims = self.types[item][i] + 1
            current_dims = img_tensor.ndim
            if current_dims < target_dims:
                for _ in range(target_dims - current_dims):
                    img_tensor = img_tensor.unsqueeze(0)
            elif current_dims > target_dims:
                img_tensor = img_tensor.squeeze()

            if self.val == 0:
                img_tensor = self.transform(img_tensor)
                
            images[f"Mod{i}"] = img_tensor

        return images, self.files["Result"][item], torch.tensor(self.files["Meta"][item]), self.types[item]

    def __len__(self):
        return self.len







