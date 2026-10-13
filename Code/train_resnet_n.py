import argparse
import os
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr
from utils.Stats import evaluate

from Supervised_Dataset import Supervised_Dataset

from transformers import ViTConfig, ViTForImageClassification
from monai.networks.nets import ViT as VIT
from monai.networks.nets import resnet18 as res3d
from Testing_class import Testing
from fold_dataset import Fold_Dataset
from sklearn.model_selection import StratifiedKFold

parser = argparse.ArgumentParser()

parser.add_argument('-batch_size', '--batch_size', default=8, type=int)
parser.add_argument('--datapath', default=None, type=str)
parser.add_argument('--savepath', default=None, type=str)
parser.add_argument('--lr', default=5e-3, type=float)
parser.add_argument('--weight_decay', default=0.01, type=float)
parser.add_argument('--num_epochs', default=50, type=int)
parser.add_argument('--seed', default=1024, type=int)
parser.add_argument('--gpu', default="0", type=str)
parser.add_argument('--resume', default=False, type=bool)
parser.add_argument('--meta', default=0, type=int)
parser.add_argument('--datasets', default=-1, type=int)
parser.add_argument('--missing', default=False, type=bool)
parser.add_argument('--mode', default="Resnet", type=str)
parser.add_argument('--statistic', default=False, type=bool)
parser.add_argument('--testing', default=False, type=bool)
parser.add_argument('--kfold', default=-1, type=int)
parser.add_argument('--data_prefix', default="/data/edu/", type=str, help="Prefix path for modality data files")

path = os.path.dirname(__file__)
args = parser.parse_args()
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ["CUDA_MODULE_LOADING"] ="LAZY"
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
torch.autograd.set_detect_anomaly(True)


from torchvision.models import resnet101
from torchvision.models import resnet18


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



class Post_fuse(nn.Module):
    def __init__(self):
        super(Post_fuse,self).__init__()
        self.model3 = VIT(in_channels=1, img_size=(128,128,128),  patch_size = (16,16), num_layers=6, num_heads=6,  hidden_size=288,  pos_embed_type='sincos', classification =True, num_classes = 256)
        self.model2 = VIT(in_channels=1, img_size=(128,128,128),  patch_size = (16,16,16), num_layers=6, num_heads=6,  hidden_size=288,  pos_embed_type='sincos', classification =True, num_classes = 256)
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
        
class Glauc(nn.Module):
    def __init__(self):
        super(Glauc,self).__init__()
        self.conv2 = resnet18().to(torch.bfloat16)
        self.conv3 = res3d(n_input_channels=1, num_classes = 512).to(torch.bfloat16)
        self.final_layer = nn.Linear(1024, 256).to(torch.bfloat16)
        self.conv2.fc = nn.Linear(512, 512).to(torch.bfloat16)
        self.conv2.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False).to(torch.bfloat16)
        self.conv2 = self.conv2.to(torch.bfloat16)

    
    def forward(self,x):
        d2 = x["Mod0"].to(torch.bfloat16)
        d3 = x["Mod1"].to(torch.bfloat16)
        d2 = self.conv2(d2)
        d3 = self.conv3(d3)
        final = torch.concat([d2,d3],dim=1)
        return self.final_layer(final)


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
            self.model = ViTForImageClassification(config=self.config).to(torch.bfloat16)
        elif mode == "mmF":
            self.model = mmFuse()
        elif mode == "mmF3":
            self.model = mmFuse3d()
        elif mode == "Glauc":
            self.model = Glauc().to(torch.bfloat16)
            
        self.meta = meta
        if meta and mode == "Glauc":
            self.final_layer = nn.Linear(256+meta, classes).to(torch.bfloat16)
        elif meta:
            self.final_layer = nn.Linear(256+meta, classes).to(torch.float32)
        else:
            self.final_layer = nn.Linear(256, classes).to(torch.bfloat16)


    def forward(self,x,in_meta):
        if self.mode !="mmF" and self.mode !="mmF3" and self.mode !="Glauc":
            x = x["Mod0"].to(torch.bfloat16)
        x = self.model(x)
        if self.mode =="VIT":
            x = x.logits
        stored_type = x.dtype
        if self.meta:
            x = torch.cat([x,in_meta],dim=1)
            
            x = self.final_layer(x)
            
        x = x.to(dtype=stored_type)
            
        return x

ckpts = args.savepath

os.makedirs(ckpts, exist_ok=True)

criterion = nn.CrossEntropyLoss()

def cal_loss(pred, y):
    final_loss = criterion(pred, y)
    return final_loss

def save_model(model, epoch, auc, path, optim):
    print("Saved with auc:" + str(auc) + " at the iteration " + str(epoch))
    torch.save({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optim_dict': optim.state_dict(),
        'best': auc
    },
        os.path.join(path, 'train_best.pth'))
    

def dict_to_tensor(dict):
    for key, _ in dict.items():
        dict[key] = dict[key].cuda(non_blocking = True).to(torch.float32)
    return dict

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

    '''for i in range(dd_max):
        final_transformer["Mod"+str(i)] = torch.stack([x["Mod"+str(dd[index][i])] for index,x in enumerate(transformed) if len(dd[index])>i])

    for i in range(td_max):
        final_transformer["Mod"+str(i+dd_max)] = torch.stack([x["Mod"+str(td[index][i])]  for index,x in enumerate(transformed) if len(td[index])>i])'''
        

    final_types = list(map(lambda x: "".join(map(str,sorted(x))),types))
    final_types = np.stack(final_types,axis=0)
    final_meta = torch.stack(meta,dim=0)
    final_y = torch.stack([torch.tensor(a) for a in y],axis=0)

    return  final_images, final_y, final_meta, final_types




def kf_main():
    s=1024
    initial_ckpts=args.savepath
    models=[ "Glauc"]
    sets = [1]
    
    n_mod=0
    
    if args.mode !="Multi":
        models = [args.mode]
    
    os.makedirs(initial_ckpts, exist_ok=True)
    # Set seeds
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    random.seed(s)
    np.random.seed(s)
    
    cudnn.benchmark = False
    cudnn.deterministic = True

    def seed_worker(worker_id):
        worker_seed = s
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(s)
    
    if args.mode !="Multi":
        models = [args.mode]
    for mod in models:
        n_mod+=1
        full_dataset = Fold_Dataset(root=args.datapath, meta=5, datasets=sets[n_mod-1], miss_mod= args.missing, data_prefix=args.data_prefix)
        strat_keys = np.array([f"{m[-1]}_{r}" for m, r in zip(full_dataset.files["Meta"], full_dataset.files["Result"])])
        indices = np.arange(len(strat_keys))
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
        
        os.makedirs(os.path.join(initial_ckpts,mod), exist_ok=True)
    
        for fold, (train_idx, test_idx) in enumerate(skf.split(indices, strat_keys)):
            
            if fold>=args.kfold:
                
                best = 0
                
                ckpts = os.path.join(initial_ckpts,mod,f"n_fold_{fold}")
                
                os.makedirs(ckpts, exist_ok=True)
                
                if args.resume==True:
                    log_file = open(os.path.join(ckpts, 'logging.txt'), 'a')
                else:
                    log_file = open(os.path.join(ckpts, 'logging.txt'), 'w+')

                # Set cuda parameter

                start = time.time()
                print(f"\n{'='*20} ROUND {fold + 1}/5 {'='*20}")
                
                full_dataset.set_indices(train_idx)
                full_dataset.val = 0 # Training mode (Augmentation ON)
                train_loader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=True, collate_fn = _custom_collate_fn, persistent_workers=True, num_workers=12)
                
                test_dataset = Fold_Dataset(args.datapath, meta=args.meta, datasets=sets[n_mod-1], miss_mod=args.missing, data_prefix=args.data_prefix)
                test_dataset.set_indices(test_idx)
                test_dataset.val = 1 
                
                init_epoch = 0
                

                # Model + hyperparameters
                model = Model(args.meta,full_dataset.max_lab(), mode=mod).cuda().to(torch.bfloat16)
            

                opt = {"resume":args.resume,"tasks": full_dataset.obtain_tasks(), "path": ckpts,  "b_size": args.batch_size, "weight_decay": args.weight_decay, "lr": args.lr, "TrainLoader": train_loader}
                secondary  =  [evaluate() for _ in range(opt["tasks"])]
                train_stats = evaluate()
                train_params = [{'params': model.parameters() , 'lr': opt["lr"],
                                    'weight_decay': opt["weight_decay"]}]
                optimizer = torch.optim.Adam(train_params, betas=(0.9, 0.999), eps=1e-08, amsgrad=True)
                lr_schedule = lr.CosineAnnealingLR( optimizer=optimizer,
                                                    T_max=(args.num_epochs) * len(train_loader))
                
                if args.resume == True and os.path.exists(os.path.join(ckpts, 'train_best.pth')):
                    checkpoint = torch.load(os.path.join(ckpts, 'train_best.pth'), weights_only=False)
                    model.load_state_dict(checkpoint['state_dict'], strict=False)
                    init_epoch = checkpoint['epoch']-20
                
                best =0
                num_params = 0
                for param in model.parameters():
                    num_params += param.numel()
                print('Total number of parameters: %d' % num_params)
                for epoch in range(init_epoch+1, args.num_epochs):
                    train_iter = iter(train_loader)
                    vectors = [{}for _ in range(0,full_dataset.max_lab())]
                    model.train()
                    model.is_training = True
                    iter_per_epoch = len(train_loader)
                    loss_total=0
                    loss = 0
                    for i in range(iter_per_epoch):
                        try:
                            data = next(train_iter)
                        except:
                            train_iter = iter(train_loader)
                            data = next(train_iter)
                        images, y, meta, _ = data
                        meta = torch.as_tensor(meta).to(torch.float32).cuda(non_blocking=True)
                        images = dict_to_tensor(images)
                        targets = y
                        y = torch.FloatTensor([[int(i==num) for num in range(0,full_dataset.max_lab())] for i in y])
                        y = y.cuda(non_blocking=True)
                        fuse_pred = model(images, meta)
                        loss_total += cal_loss(fuse_pred, y)


                        for batch,val in enumerate(fuse_pred.cpu().detach()):
                            secondary[int(meta[batch][-1])].update(val.unsqueeze(0), targets[batch].cpu().to(torch.uint8).unsqueeze(0), loss_total.item())
                        train_stats.update(fuse_pred.cpu().detach(), targets.cpu().to(torch.uint8), loss_total.item())
                        if i%10==0:
                            msg = 'Train for Epoch {}/{}, Iter {}/{}, Loss {:.4f}, Acc {:.4f}, Task {:d}, Auc task {:.4f}, Acc task {:.4f}'.format((epoch + 1),
                                                                                                                    args.num_epochs,
                                                                                                                    (i + 1),
                                                                                                                    iter_per_epoch,
                                                                                                                    train_stats.avg_loss(),
                                                                                                                    train_stats.total_acc(),
                                                                                                                    i%len(secondary),
                                                                                                                    secondary[i%len(secondary)].auc_score(),
                                                                                                                    secondary[i%len(secondary)].total_acc())
                            print(msg)
                        loss_total.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        lr_schedule.step()
                        loss_total=0.
                        
                    msg = "Train for Epoch {}/{}, Average Loss {:.4f}, Total Acc {:.4f}".format(
                        (epoch + 1), args.num_epochs, train_stats.avg_loss(), train_stats.total_acc())
                    msg = msg + " ".join(["Acc task {} {:.4f}".format(s,secondary[s].total_acc()) for s in range(len(secondary))]) + "\n"
                    print(msg)
                    log_file.write(msg)
                    log_file.flush()
                    vectors = vectors

                        
                    if best< train_stats.total_acc():
                        best = train_stats.total_acc()
                        save_model(model,epoch, train_stats.total_acc(), ckpts, optimizer)
                        msg = "Saved model at epoch{:d} with accuracy {:.4f}\n".format(
                            (epoch + 1), train_stats.total_acc())
                        print(msg)
                        msg = 'total time: {:.4f} hours'.format((time.time() - start) / 3600)
                        print(msg)
                        log_file.write(msg + "\n\n\n")
                        log_file.flush()
                        
                    train_stats.new_epoch()
                    
                    for m in range(len(secondary)):
                        secondary[m].new_epoch()
                     
                if args.testing:
                    log_file = open(os.path.join(ckpts, 'logging_test.txt'), 'w+')
                    test = Testing(model,log_file,test_dataset,1, s, ckpts)
                    test.test(True)
            

if __name__ == '__main__':
    kf_main()



