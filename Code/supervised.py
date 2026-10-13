import torch
from utils.utils import cosine_loss_supervsed, cosine_distance,plot_grad_flow
import torch.optim.lr_scheduler as lr
import torch.nn as nn
from utils.Stats import evaluate
import os
import numpy as np
from torchviz import make_dot
from model_semi import Model
from utils.utils import FocalLoss

feature_size = 768
dictionary_size = 10

def check_grad_norm(model, name):
    total_norm = 0
    for n, p in model.named_parameters():
        if p.grad is not None:
            print(f"Gradient Norm for {n}: {p:.6f}")
    

class Supervised():
    def __init__(self, model, epochs, opt, args, retain=1) -> None:
        self.model = model.to(torch.bfloat16).cuda()
        torch.compile(self.model, mode="max-autotune")
        self.epochs = epochs
        self.acumulations =args.accumulate
        self.opt = opt
        self.mod = 0
        self.last_best = 0
        self.args = args
        self.dist = opt["dist"]
        self.retain = retain
        self.init_epoch = 0
        self.path = opt["path"]
        self.log_file = open(os.path.join(opt["path"], 'logging.txt'), 'a')

        self.train_loader = opt["TrainLoader"]
        self.val_loader = opt["ValLoader"]
        self.tasks = opt["tasks"]

        self.iter_per_epoch = len(self.train_loader)
        self.train_iter = iter(self.train_loader)
        self.train_stats = evaluate()

        self.iter_val = len(self.val_loader)
        self.val_iter = iter(self.val_loader)
        self.val_stats = evaluate()
        self.secondary  =  [evaluate() for _ in range(opt["tasks"])]
        self.tertiary  =  [evaluate() for _ in range(opt["tasks"])]
        self.vectors  =  self.vectors = torch.zeros((opt["tasks"], dictionary_size, feature_size))
        
        bert_params = [p for n, p in self.model.named_parameters() if "bert" in n]
        other_params = [p for n, p in self.model.named_parameters() if "bert" not in n]

        self.optimizer = torch.optim.AdamW([{'params': bert_params, 'lr': self.opt["lr"]*0.1}, {'params': other_params, 'lr': self.opt["lr"]}], weight_decay= self.opt["weight_decay"])
        
        self.lr_schedule = lr.CosineAnnealingLR(optimizer=self.optimizer, T_max= self.iter_per_epoch* self.epochs)
        self.criterion = nn.CrossEntropyLoss()
        self.second_crit = FocalLoss()

        '''self.train_params = [{'params': list(self.model.embeddings.parameters()) + list(self.model.bert.parameters()) + list(self.model.ff.parameters()) , 'lr': self.opt["lr"],
                              'weight_decay': self.opt["weight_decay"]}]
        self.optimizer = torch.optim.SGD(self.train_params)
        self.lr_schedule = lr.CosineAnnealingLR(verbose=True, optimizer=self.optimizer, T_max= self.iter_per_epoch* self.epochs)
        self.criterion = nn.CrossEntropyLoss()
        self.train_params_enc = [{'params': list(self.model.models.parameters()) + list(self.model.trainer.parameters()) , 'lr': 5e-5,
                              'weight_decay': self.opt["weight_decay"]}]
        self.optimizer_enc = torch.optim.Adam(self.train_params_enc)
        self.lr_schedule_enc = lr.CosineAnnealingLR(verbose=True, optimizer=self.optimizer_enc, T_max= self.iter_per_epoch* self.epochs)'''

    def cal_loss(self, pred, y):
        final_loss= self.criterion(pred,y) 
        return final_loss
    
    def dict_to_tensor(self,dict):
        for key, _ in dict.items():
            dict[key] = dict[key].cuda(non_blocking = True).to(torch.bfloat16)
        return dict
    
    

    def generate_vectors(self, n_labels):
        #vectors = [[] for _ in range(0,n_labels)]
        count=0
        vectors = [{}for _ in range(0,n_labels)]
        
        with torch.no_grad():
            self.train_iter = iter(self.train_loader)
            for _ in range(self.iter_per_epoch):
                try:
                    data = next(self.train_iter)
                except:
                    self.train_iter = iter(self.train_loader)
                    data = next(self.train_iter)
                images, y, meta, types = data
                images = self.dict_to_tensor(images)
                y = y.cpu().to(torch.uint8)
                meta = torch.as_tensor(meta).to(torch.bfloat16).cuda(non_blocking=True)
                to_get = []
                for i,type in enumerate(meta[:,-1]):
                    type = int(type)
                    res = str(y[i].item())
                    if res not in vectors[type]:
                        vectors[type][res]= torch.Tensor([])
                    if len(vectors[type][res])<dictionary_size:
                        to_get.append(i)

                if len(to_get)>0:
                    _,fuse_pred,_ = self.model(images,meta, types)
                    for i in to_get:
                        if len(vectors[int(meta[i,-1])][str(y[i].item())]) < dictionary_size:
                            vectors[int(meta[i,-1])][str(y[i].item())] = torch.cat([vectors[int(meta[i,-1])][str(y[i].item())],fuse_pred[i].detach().cpu().unsqueeze(0)],dim=0)
                            count+=1

            self.vectors = vectors

        return self.vectors

    def save_model(self, epoch, auc):
        msg = "Saved with auc:{:.4f} at the iteration\n".format(auc)
        self.log_file.write(msg)
        self.log_file.flush()
        print("Saved with auc:" + str(auc) + " at the iteration " + str(epoch))
        torch.save({
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optim_dict': self.optimizer.state_dict(),
            'best': auc,
            'args':self.args
        },
            os.path.join(self.opt["path"], 'train_best.pth'))

    def load(self, encoder=False):
        checkpoint = torch.load(os.path.join(self.path, 'train_best.pth'))
        self.model.load_state_dict(checkpoint['state_dict'], strict=False)
        self.optimizer.load_state_dict(checkpoint['optim_dict'])
        self.val_stats.set_best_acc(checkpoint['best'], checkpoint['epoch'])
        self.init_epoch = checkpoint['epoch']
        return self.model

    def apply_along_axis(self, function, x, axis=0):
        return torch.stack([
            function(x_i) for x_i in torch.unbind(x, dim=axis)
        ], dim=axis)

    def train(self, n_epochs, label):
        self.max_label = label
        if self.opt["resume"]==True:
            self.load()
        if self.opt["modifications"][1]==1:
            self.generate_vectors(self.tasks)
        for epoch in range(self.init_epoch, n_epochs):

            if epoch%10==0:
                vectors = [{}for _ in range(0,self.tasks)]
            self.model.train()
            self.model.is_training = True
            for name, param in self.model.named_parameters():
                if "classifier" in name:
                    print(f"{name} is trainable: {param.requires_grad}")

            self.train_iter = iter(self.train_loader)
            loss = 0
            for i in range(self.iter_per_epoch):
                try:
                    data = next(self.train_iter)
                except:
                    self.train_iter = iter(self.train_loader)
                    data = next(self.train_iter)
                images, y, meta, type = data
                meta = torch.as_tensor(meta).to(torch.bfloat16).cuda(non_blocking=True)
                images = self.dict_to_tensor(images)
                targets = y
                y_long = y.to(torch.int64) # one_hot requires long/int64
                y = torch.nn.functional.one_hot(y_long, num_classes=self.max_label).to(torch.bfloat16).cuda(non_blocking=True)
                
                    
                fuse_pred, obt_vect, helpers = self.model(images, meta, type)
            #exit()
            #make_dot(fuse_pred.mean(), params=dict(self.model.named_parameters()), show_attrs=True, show_saved=True).render(directory='/data/edu/PETCT', view=True)
                
                loss_total = self.cal_loss(fuse_pred, y)
                
                if self.opt["modifications"][1]==1:
                    if epoch >10:
                        loss_total += cosine_loss_supervsed(obt_vect,self.vectors,targets, meta[:,-1].to(torch.uint8))
                    
                    if epoch%10==0:
                        targets = targets.to(torch.uint8)
                        for z,type in enumerate(meta[:,-1]):
                            type = int(type)
                            res = str(targets[z].item())
                            obtained = obt_vect[z].detach()
                            if res not in vectors[type]:
                                vectors[type][res]= torch.empty((0, obtained.shape[0]), device=obtained.device, dtype=obtained.dtype)
                            if vectors[type][res].shape[0]<dictionary_size:
                                current_dict = vectors[type][res]
                                if current_dict.shape[0] > 0 and current_dict.shape[1] != obtained.shape[0]:
                                    diff = current_dict.shape[1] - obtained.shape[0]
                                    if diff < 0:
                                        vectors[type][res] = torch.nn.functional.pad(current_dict, (0, abs(diff)))
                                    else:
                                        obtained = torch.nn.functional.pad(obtained, (0, diff))
                                        
                                vectors[type][res] = torch.cat([vectors[type][res], obtained.unsqueeze(0)], dim=0)
                
                if self.opt["modifications"][0]==1:
                    loss_total += self.criterion(helpers, y)*0.3
                    
                for batch,val in enumerate(fuse_pred.cpu().detach()):
                    self.secondary[int(meta[batch][-1])].update(val.unsqueeze(0), targets[batch].cpu().to(torch.uint8).unsqueeze(0), loss_total.item()) 
                self.train_stats.update(fuse_pred.cpu().detach(), targets.cpu().to(torch.uint8), loss_total.item())
                
                if i%10==0:
                    msg = 'Train for Epoch {}/{}, Iter {}/{}, Loss {:.4f}, Acc {:.4f}, Task {:d}, Auc task {:.4f}, Acc task {:.4f}'.format((epoch + 1),
                                                                                                        n_epochs,
                                                                                                        (i + 1),
                                                                                                        self.iter_per_epoch,
                                                                                                        self.train_stats.avg_loss(),
                                                                                                        self.train_stats.total_acc(),
                                                                                                        i%len(self.secondary),
                                                                                                        self.secondary[i%len(self.secondary)].auc_score(),
                                                                                                        self.secondary[i%len(self.secondary)].total_acc())
                    print(msg)
                loss_total /= self.acumulations
                loss_total.backward() 
                #check_grad_norm(self.model.models["2"], "Encoder_2")
                #check_grad_norm(self.model.embeddings, "Embeddings_Proj")
                
                
                if (i+1) % self.acumulations ==0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                self.lr_schedule.step()
                    
                    
                
            msg = "Train for Epoch {}/{}, Average Loss {:.4f}, Total Acc {:.4f}".format(
                (epoch + 1), self.epochs, self.train_stats.avg_loss(), self.train_stats.total_acc())
            msg = msg + " ".join(["Acc task {} {:.4f}".format(s,self.secondary[s].total_acc()) for s in range(len(self.secondary))]) + "\n"
            print(msg)
            self.log_file.write(msg)
            self.log_file.flush()
            
            if self.opt["modifications"][1]==1 and epoch%10==0:
                all_full = True
                for k in vectors:
                    for val in k.keys():
                        if len(k[val])<dictionary_size:
                            all_full=False
                            break
                    
                if all_full:
                    self.vectors = vectors

            with torch.no_grad():
                self.model.eval()
                self.model.is_training = False
                for i in range(self.iter_val):
                    try:
                        data = next(self.val_iter)
                    except:
                        self.val_iter = iter(self.val_loader)
                        data = next(self.val_iter)
                    images, y, meta, type = data
                    meta = torch.as_tensor(meta).to(torch.bfloat16).cuda(non_blocking=True)
                    targets = y
                    y_long = y.to(torch.int64) # one_hot requires long/int64
                    y = torch.nn.functional.one_hot(y_long, num_classes=self.max_label).to(torch.bfloat16).cuda(non_blocking=True)
                    images = self.dict_to_tensor(images)
                    fuse_pred, obt_vect,_ = self.model(images, meta,type, val = True)
                    loss = self.cal_loss(fuse_pred, y)
                    self.val_stats.update(fuse_pred.cpu().detach(), targets.cpu().to(torch.uint8), loss.item())
                    for batch,val in enumerate(fuse_pred.cpu().detach()):
                        self.tertiary[int(meta[batch][-1])].update(val.unsqueeze(0), targets[batch].cpu().to(torch.uint8).unsqueeze(0), loss.item())

                    msg = 'Val for Epoch {}/{}, Iter {}/{}, Loss {:.4f}, Acc {:.4f}, Task {:d}, Acc task {:.4f},'.format((epoch + 1),
                                                                                                      n_epochs,
                                                                                                      (i + 1),
                                                                                                      len(self.val_loader),
                                                                                                      self.val_stats.avg_loss(),
                                                                                                      self.val_stats.total_acc(),
                                                                                                      i%len(self.secondary),
                                                                                                      self.tertiary[i%len(self.secondary)].total_acc())
                    print(msg)

                msg = "Val for Epoch {}/{}, Average Loss {:.4f}, Total Acc {:.4f}".format(
                    (epoch + 1), self.epochs, self.val_stats.avg_loss(), self.val_stats.total_acc())
                msg = msg + " ".join(["Acc task {} {:.4f}".format(s,self.tertiary[s].total_acc()) for s in range(len(self.tertiary))]) + "\n"
                print(msg)
                self.log_file.write(msg)
                self.log_file.flush()

            acc_mean = np.mean([self.tertiary[i].total_acc() for i in range(0,len(self.secondary))])
            self.train_stats.new_epoch()
            self.val_stats.new_epoch()
            for m in range(len(self.tertiary)):
                self.tertiary[m].new_epoch()
            
            for m in range(len(self.secondary)):
                self.secondary[m].new_epoch()

            print(acc_mean)
            if self.last_best< acc_mean:
                self.last_best = acc_mean
                self.save_model(epoch, acc_mean)
                msg = "Saved model at epoch{:d} with accuracy {:.4f}\n".format(
                    (epoch + 1), self.val_stats.total_acc())
                print(msg)

    def start(self, label):
        # First part of the training
        self.max_label = label
        torch.set_grad_enabled(True)
        self.model.is_training = True
        self.model.train()
        # self.model.freeze_encoder()
        self.train(self.epochs)

