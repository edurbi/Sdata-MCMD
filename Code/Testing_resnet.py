import argparse
import os
import time
import logging

import numpy as np
import pandas as pd

import seaborn


from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim

import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import scikitplot as skplt
from sklearn.metrics import roc_curve, auc, classification_report



from utils.Stats import evaluate

from Supervised_Dataset import Supervised_Dataset

from transformers import ViTConfig, ViTForImageClassification
from monai.networks.nets import ViT as VIT
from monai.networks.nets import resnet18 as res3d




parser = argparse.ArgumentParser()

parser.add_argument('-batch_size', '--batch_size', default=1, type=int, help='Batch size')
parser.add_argument('--datapath', default=None, type=str)
parser.add_argument('--savepath', default=None, type=str)
parser.add_argument('--resume', default=None, type=str)
parser.add_argument('--gpu', default="0", type=str)
parser.add_argument('--n_cross', default=5, type=int)
parser.add_argument('--meta', default=5, type=int)
parser.add_argument('--mode', default="Resnet", type=str)
parser.add_argument('--datasets', default=-1, type=int)
parser.add_argument('--missing', default=False, type=bool)
parser.add_argument('--data_prefix', default="/data/edu/", type=str, help="Prefix path for modality data files")
path = os.path.dirname(__file__)
args = parser.parse_args()
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

from torchvision.models import resnet101 

class mmFuse(nn.Module):
    def __init__(self):
        super(mmFuse,self).__init__()
        self.model3 = VIT(in_channels=1, img_size=(128,128,128), patch_size = (16,16,16),  num_layers=6, num_heads=6,  hidden_size=72, pos_embed_type='sincos')
        self.model2 = VIT(in_channels=1, img_size=(256,256),  spatial_dims=2, patch_size = (16,16), num_layers=6, num_heads=6,  hidden_size=144,  pos_embed_type='sincos')
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=7, padding=3)
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=7, padding=3)
        self.conv2_1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, padding=3)
        self.conv3_1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, padding=3)
        self.relu = torch.nn.ReLU()
        self.classifier = VIT(in_channels=2, img_size=(256,144),  spatial_dims=2, patch_size = (16,16), num_layers=6, num_heads=6,  hidden_size=144,  pos_embed_type='sincos', classification =True, num_classes = 256)


    
    def forward(self,x):
        d2 = x["Mod0"].to(torch.float32)
        d3 = x["Mod1"].to(torch.float32)
        d2,_ = self.model2(d2)
        d3,_ = self.model3(d3)
        b,x,y = d2.shape
        d2 = torch.unsqueeze(d2, dim =1)
        
        d3 = torch.unsqueeze(d3, dim =1)
        d3 = d3.view(b,1,x,y)

        concated = torch.cat([d2,d3], dim =1)

        d2 = self.conv2(d2)
        d3 = self.conv3(d3)

        d4 = self.relu(d2 + d3)*concated

        d5 = self.conv2_1(d4)
        d6 = self.conv3_1(d4)

        d7 = self.relu(d5 + d6)*d4
        d7,_ = self.classifier(d7)

        return d7
    

class Glauc(nn.Module):
    def __init__(self):
        super(Glauc,self).__init__()
        self.conv2 = res3d(num_classes = 256, n_input_channels=1, spatial_dims=2 )
        self.conv3 = res3d(n_input_channels=1, num_classes = 256)
        self.pool = nn.AvgPool1d(kernel_size=2)

    
    def forward(self,x):
        d2 = x["Mod0"].to(torch.float32)
        d3 = x["Mod1"].to(torch.float32)
        d2 = self.conv2(d2)
        d3 = self.conv3(d3)
        d2 = self.pool(d2)
        d3 = self.pool(d3)
        final = torch.concat([d2,d3],dim=1)
        return final

class mmFuse3d(nn.Module):
    def __init__(self):
        super(mmFuse3d,self).__init__()
        self.model3 = VIT(in_channels=1, img_size=(128,128,128), patch_size = (16,16,16),  num_layers=6, num_heads=6,  hidden_size=288, pos_embed_type='sincos')
        self.model2 = VIT(in_channels=1, img_size=(128,128,128),   patch_size = (16,16,16), num_layers=6, num_heads=6,  hidden_size=288,  pos_embed_type='sincos')
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=7, padding=3)
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=7, padding=3)
        self.conv2_1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, padding=3)
        self.conv3_1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, padding=3)
        self.relu = torch.nn.ReLU()
        self.classifier = VIT(in_channels=2, img_size=(512,288),  spatial_dims=2, patch_size = (16,16), num_layers=6, num_heads=6,  hidden_size=288,  pos_embed_type='sincos', classification =True, num_classes = 256)


    
    def forward(self,x):
        d2 = x["Mod0"].to(torch.float32)
        d3 = x["Mod1"].to(torch.float32)
        d2,_ = self.model2(d2)
        d3,_ = self.model3(d3)
        d2 = torch.unsqueeze(d2, dim =1)
        
        d3 = torch.unsqueeze(d3, dim =1)

        concated = torch.cat([d2,d3], dim =1)

        d2 = self.conv2(d2)
        d3 = self.conv3(d3)

        d4 = self.relu(d2 + d3)*concated

        d5 = self.conv2_1(d4)
        d6 = self.conv3_1(d4)

        d7 = self.relu(d5 + d6)*d4
        d7,_ = self.classifier(d7)

        return self.relu(d7)
        


class Model(nn.Module):
    def __init__(self, meta = False, classes = 4, mode = "Resnet"):
        super(Model, self).__init__()
        self.mode = mode
        if mode == "Resnet":
            self.model = resnet101(num_classes = (256 if meta else classes)).to(torch.bfloat16)
            self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False).to(torch.bfloat16)
        elif mode == "VIT":
            self.config = ViTConfig(image_size = 256, num_channels=1, num_labels  =(256 if meta else classes) )
            self.model = ViTForImageClassification(config=self.config).to(torch.float32)
        elif mode == "mmF":
            self.model = mmFuse()
        elif mode == "mmF3":
            self.model = mmFuse3d()
        elif mode == "Glauc":
            self.model = Glauc()
            
        self.meta = meta
        if meta:
            self.final_layer = nn.Linear(256+meta, classes).to(torch.float32)

    def forward(self,x,in_meta):
        if self.mode !="mmF" and self.mode !="mmF3" and self.mode !="Glauc":
            x = x["Mod0"]
        x = self.model(x)
        if self.mode =="VIT":
            x = x.logits
        x = torch.cat([x,in_meta],dim=1)
        if self.meta:
            x = self.final_layer(x)
        return x

ckpts = args.savepath

os.makedirs(ckpts, exist_ok=True)

log_file = open(os.path.join(ckpts, 'logging_test.txt'), 'w+')



def _custom_collate_fn(batch):
    final_images = {}
    images, y ,meta, types = zip(*batch)
    td = list(map(lambda x: [int(index) for index,letter in enumerate(x) if letter==3],types))
    dd = list(map(lambda x: [int(index) for index,letter in enumerate(x) if letter==2],types))
    dd_max = max([len(q) for q in dd])
    td_max = max([len(q) for q in td])
    for i in range(dd_max):
        final_images["Mod"+str(i)] = torch.stack([x["Mod"+str(dd[index][i])] for index,x in enumerate(images) if len(dd[index])>i])

    for i in range(td_max):
        final_images["Mod"+str(i+dd_max)] = torch.stack([x["Mod"+str(td[index][i])] for index,x in enumerate(images) if len(td[index])>i])

    final_types = list(map(lambda x: "".join(map(str,sorted(x))),types))
    final_types = np.stack(final_types,axis=0)
    final_meta = torch.stack(meta,dim=0)
    final_y = torch.stack([torch.tensor(a) for a in y],axis=0)

    return  final_images, final_y, final_meta, final_types


def dict_to_tensor(dict):
    for key, _ in dict.items():
        dict[key] = dict[key].cuda(non_blocking = True).to(torch.float32)
    return dict

saves=[{"Pred":np.array([]),"Lab":np.array([])} for _ in range(0,5)]


def main(p):
    val_set = Supervised_Dataset(args.datapath, 2, args.meta, datasets=args.datasets, miss_mod=args.missing, data_prefix=args.data_prefix)
    val_loader = torch.utils.data.DataLoader(
        dataset=val_set,
        batch_size=args.batch_size,
        num_workers=10,
        shuffle=True,
        pin_memory=False,
        collate_fn = _custom_collate_fn)
    ##########setting seed
    cudnn.deterministic = True
    torch.set_float32_matmul_precision('high')

    ##########setting models
    model = Model(args.meta,val_set.max_lab(), mode=args.mode).cuda()
    torch.compile(model, mode="max-autotune")
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print('Total number of parameters: %d' % num_params)

    ##########Setting data
    val_stats = evaluate()
    logging.info(str(args))
    val_iter = iter(val_loader)


    if args.resume is not None:
        checkpoint = torch.load(args.resume+"/train_best.pth")
        logging.info('best epoch: {}'.format(checkpoint['epoch']))
        model.load_state_dict(checkpoint['state_dict'], strict=False)       


    start = time.time()
    logging.info('#############Testing############')
    msg = "Starting time: {:.1f}\n".format(start)
    log_file.write(msg)
    log_file.flush()
    types = []

    criterion=nn.CrossEntropyLoss()
    tertiary  =  [evaluate() for _ in range(3)]

    res =None
    true=None
    with torch.no_grad():
        model.eval()
        model.is_training = False
        for i in range(len(val_iter)):
            try:
                data = next(val_iter)
            except:
                val_iter = iter(val_loader)
                data = next(val_iter)
            images, y, meta, type = data
            meta = torch.as_tensor(meta).to(torch.bfloat16).cuda(non_blocking=True)
            target = torch.tensor(y).to(torch.bfloat16).cuda()
            y = torch.FloatTensor([[int(i==num) for num in range(0,val_set.max_lab())] for i in y])
            y = y.to(torch.bfloat16).cuda(non_blocking=True)
            images = dict_to_tensor(images)
            fuse_pred = model(images, meta)
            loss = criterion(fuse_pred, y)
            val_stats.update(fuse_pred.cpu().detach(), target.cpu().to(torch.uint8), loss.item())
            for batch, val in enumerate(fuse_pred):
                if len(saves[int(meta[batch][-1])]["Pred"])!=0:
                    saves[int(meta[batch][-1])]["Pred"]= np.concatenate((saves[int(meta[batch][-1])]["Pred"],[fuse_pred.cpu().detach().to(torch.float16)[batch]]))
                    saves[int(meta[batch][-1])]["Lab"]= np.concatenate((saves[int(meta[batch][-1])]["Lab"],np.array([target[batch].cpu().detach().to(torch.float16)])))
                else:
                    saves[int(meta[batch][-1])]["Pred"]= np.array([fuse_pred.cpu().detach().to(torch.float16)[batch]])
                    saves[int(meta[batch][-1])]["Lab"]= np.array([target[batch].cpu().detach().to(torch.float16)])
                tertiary[int(meta[batch][-1])].update(val.unsqueeze(0), target[batch].cpu().to(torch.uint8).unsqueeze(0), loss.item())

            msg = 'Iter {}/{}, Loss {:.4f}, Acc {:.4f}, Task {:d}, Auc task {:.4f}, Acc task {:.4f}'.format(
                                                                                                      (i + 1),
                                                                                                      len(val_iter),
                                                                                                      val_stats.avg_loss(),
                                                                                                      val_stats.total_acc(),
                                                                                                      i%len(tertiary),
                                                                                                      tertiary[i%len(tertiary)].auc_score(),
                                                                                                      tertiary[i%len(tertiary)].total_acc())
            print(msg)


        msg = "Average Loss {:.4f}, Total Acc {:.4f}".format(
                 val_stats.avg_loss(), val_stats.total_acc())
        msg = msg + " ".join([" Acc task {} {:.4f}, Auc {:.4f}".format(s,tertiary[s].total_acc(), tertiary[s].auc_score()) for s in range(len(tertiary))]) + "\n"
        print(msg)
        log_file.write(msg)
        log_file.flush()
    for q in range(0,(3 if args.datasets==-1 else 1)):
        if q!=1:
            print(saves[q]["Lab"])
            print(saves[q]["Pred"])
            
            # Calculate ROC curve
            ''' fpr, tpr, thresholds = roc_curve(saves[q]["Lab"], saves[q]["Pred"][:,1]) 
            roc_auc = auc(fpr, tpr)
            # Plot the ROC curve
            plt.figure()  
            plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve for Breast Cancer Classification')
            plt.legend()
            plt.savefig(ckpts+"/curve"+str(q)+".png")
            plt.clf()'''
        print(saves[q]["Pred"])
        msg= classification_report(saves[q]["Lab"].astype(int), np.argmax(saves[q]["Pred"], axis=1).astype(int))
        log_file.write(msg)
        log_file.flush()



if __name__ == '__main__':
    for i in range(0,1):
        main(i)







from sklearn.metrics import precision_recall_curve, f1_score
'''#precisions, recalls, thresholds = precision_recall_curve(label_results['actual'].values, label_results['predicted'].values)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)

optimal_idx = np.argmax(f1_scores)
th = thresholds[optimal_idx]
optimal_f1 = f1_scores[optimal_idx]

print(f'Optimal Threshold: {th}, F1 Score: {optimal_f1}')'''