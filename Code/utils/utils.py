import numpy as np
import torch
from torch.nn import Softmax, CosineSimilarity
from torch import norm
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from numba import jit



class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2,reduction='mean'):
        super(FocalLoss, self).__init__(weight,reduction=reduction)
        self.gamma = gamma
        self.weight = weight 

    def forward(self, input, target):

        ce_loss = F.cross_entropy(input, target,reduction=self.reduction,weight=self.weight) 
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss
    
def cosine_loss_supervsed(tensors,bases,y, tasks, size=256):
    device = tensors.device
    dim = tensors.shape[-1]
    batch_size = tensors.shape[0]
    
    tensors_norm = torch.nn.functional.normalize(tensors, p=2, dim=1)
    centroids = {}
    for q, task_dict in enumerate(bases):
        centroids[q] = {}
        for label_str, class_tensor in task_dict.items():
            if class_tensor.shape[0] > 0:
                mean_vec = torch.mean(class_tensor[:, :dim].to(device), dim=0).to(tensors.dtype)
                centroids[q][label_str] = torch.nn.functional.normalize(mean_vec, p=2, dim=0)

    final_loss = 0.0
    valid_samples = 0

    for i in range(batch_size):
        curr_vec = tensors_norm[i]
        curr_task = tasks[i].item()
        curr_label = str(y[i].item())
        
        loss_selected = 0.0 # Push (Different tasks)
        loss_equal = 0.0    # Pull (Same task, same class)
        
        num_diff_tasks = 0
        
        # Iterate through our calculated centroids
        for q, class_centroids in centroids.items():
            for label_str, mean_vec_norm in class_centroids.items():
                mean_vec_norm.to(torch.bfloat16)
                # Similarity between current sample and the class mean
                similarity = torch.dot(curr_vec, mean_vec_norm)

                if q != curr_task:
                    # PUSH: We want similarity to other tasks to be 0 (or less)
                    loss_selected += torch.relu(similarity) 
                    num_diff_tasks += 1
                elif label_str == curr_label:
                    # PULL: We want similarity to the same class mean to be 1.0
                    loss_equal += torch.relu(1.0 - similarity)

        # Average the push/pull for this sample
        push_component = loss_selected / max(num_diff_tasks, 1)
        final_loss += (push_component + loss_equal) / 2.0
        valid_samples += 1

    return final_loss / max(valid_samples, 1)



def cosine_distance(tensors,bases):
    final_loss = 0
    for i in range(0,len(tensors)):
        vector = tensors[i]
        loss = {q: CosineSimilarity(dim=0) for q in range(len(bases))}
        aux = torch.stack([l(vector,bases[n]) for n,l in enumerate(loss.values())])
        aux = 1-torch.mean(aux)
        final_loss += aux
    
    final_loss = final_loss/(len(tensors)) 

    return final_loss




def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            if p.grad is not None:
                layers.append(n)
                ave_grads.append(torch.log(p.grad.abs().mean().cpu().to(torch.float16)))
    plt.figure(figsize=(15, 5))
    plt.subplots_adjust(bottom=0.55)
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.savefig("/data/edu/PETCT/Code/utils/newfig.png")