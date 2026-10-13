import torch 
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc, confusion_matrix, accuracy_score
from utils.Stats import evaluate
import torch.nn as nn
import time
from matplotlib import pyplot as plt
import random

def softmax(array):
    return np.array([np.exp(a-np.max(a)) /np.exp(a-np.max(a)).sum() for a in array])

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

def to_categorical(label):
    b = np.zeros((label.size, label.max()+1))
    b[np.arange(label.size), label] =1
    return b

class Testing():
    def __init__(self, model, log, test_set, datasets, seed, path):
        super(Testing, self).__init__()
        self.model = model
        self.log=log
        self.test_set = test_set
        g = torch.Generator()
        g.manual_seed(seed)
        self.path = path
        self.soft = nn.Softmax()
        
        def seed_worker(worker_id):
            worker_seed = seed
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            
        self.test_loader = torch.utils.data.DataLoader(
            dataset=test_set,
            batch_size=10,
            shuffle=True,
            pin_memory=False,
            worker_init_fn=seed,
            generator=g,
            collate_fn = _custom_collate_fn)
        self.datasets = datasets
        self.test_stats = evaluate()
        
        start = time.time()
        msg = "Starting time: {:.1f}\n".format(start)
        self.log.write("########Testing#########")
        self.log.flush()
        self.log.write(msg)
        self.log.flush()

        self.criterion=nn.CrossEntropyLoss()
        self.tertiary  =  [evaluate() for _ in range(3)]
    
    def test(self, special=False):
        val_iter = iter(self.test_loader)
        with torch.no_grad():
            saves=[{"Pred":np.array([]),"Lab":np.array([])} for _ in range(0,5)]
            self.model.eval()
            self.model.is_training = False
            for i in range(len(self.test_loader)):
                try:
                    data = next(val_iter)
                except:
                    val_iter = iter(self.test_loader)
                    data = next(val_iter)
                images, y, meta, type = data
                meta = torch.as_tensor(meta).to(torch.bfloat16).cuda(non_blocking=True)
                target = torch.as_tensor(y).to(torch.bfloat16).cuda()
                y = torch.FloatTensor([[int(i==num) for num in range(0,self.test_set.max_lab())] for i in y])
                y = y.to(torch.bfloat16).cuda(non_blocking=True)
                images = dict_to_tensor(images)
                if special:
                    fuse_pred= self.model(images, meta)
                else:
                    fuse_pred, obt_vect,_ = self.model(images, meta,type)
                loss = self.criterion(fuse_pred, y)
                self.test_stats.update(fuse_pred.cpu().detach(), target.cpu().to(torch.uint8), loss.item())
                for batch, val in enumerate(fuse_pred):
                    if len(saves[int(meta[batch][-1])]["Pred"])!=0:
                        saves[int(meta[batch][-1])]["Pred"]= np.concatenate((saves[int(meta[batch][-1])]["Pred"],[fuse_pred.cpu().detach().to(torch.float32)[batch]]))
                        saves[int(meta[batch][-1])]["Lab"]= np.concatenate((saves[int(meta[batch][-1])]["Lab"],np.array([target[batch].cpu().detach().to(torch.uint8)])))
                    else:
                        saves[int(meta[batch][-1])]["Pred"]= np.array([fuse_pred.cpu().detach().to(torch.float32)[batch]])
                        saves[int(meta[batch][-1])]["Lab"]= np.array([target[batch].cpu().detach().to(torch.uint8)])
                    self.tertiary[int(meta[batch][-1])].update(val.unsqueeze(0), target[batch].cpu().to(torch.uint8).unsqueeze(0), loss.item())

                msg = 'Iter {}/{}, Loss {:.4f}, Acc {:.4f}, Task {:d}, Acc task {:.4f}'.format(
                                                                                                        (i + 1),
                                                                                                        len(val_iter),
                                                                                                        self.test_stats.avg_loss(),
                                                                                                        self.test_stats.total_acc(),
                                                                                                        i%len(self.tertiary),
                                                                                                        self.tertiary[i%len(self.tertiary)].total_acc())
                print(msg)


            msg = "Average Loss {:.4f}, Total Acc {:.4f}".format(
                    self.test_stats.avg_loss(), self.test_stats.total_acc())
            msg = msg + " ".join([" Acc task {} {:.4f}".format(s,self.tertiary[s].total_acc()) for s in range(len(self.tertiary))]) + "\n"
            print(msg)
            self.log.write(msg)
            self.log.flush()
        for q in range(0,self.datasets):
            if max(saves[q]["Lab"])<2:
                
                # Calculate ROC curve
                fpr, tpr, thresholds = roc_curve(saves[q]["Lab"], saves[q]["Pred"][:,1]) 
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
                plt.savefig(self.path+"/curve"+str(q)+".png")
                plt.clf()
            msg= "Report \n"
            self.log.write(msg)
            self.log.flush()
            msg= classification_report(saves[q]["Lab"].astype(int), np.argmax(saves[q]["Pred"], axis=1).astype(int), digits=4)
            self.log.write(msg)
            self.log.flush()
            msg= "Confusion matrix \n"
            self.log.write(msg)
            self.log.flush()
            msg= str(confusion_matrix(saves[q]["Lab"].astype(int), np.argmax(saves[q]["Pred"], axis=1).astype(int)))
            self.log.write(msg)
            self.log.flush()
            msg= "\n Total AUC is " + str(roc_auc_score(to_categorical(saves[q]["Lab"].astype(int)), softmax(saves[q]["Pred"][:,0:max(saves[q]["Lab"])+1]),  multi_class='ovo'))
            self.log.write(msg)
            self.log.flush()
            msg= "\n Total ACC is " + str(accuracy_score(saves[q]["Lab"].astype(int), np.argmax(saves[q]["Pred"], axis=1).astype(int)))
            self.log.write(msg)
            self.log.flush()