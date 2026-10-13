import argparse
import os
import time
import random
import numpy as np

from Testing_class import Testing

from Supervised_Dataset import Supervised_Dataset
from fold_dataset import Fold_Dataset
from supervised import Supervised
from supervised_fold import Supervised as Supervised_fold
from sklearn.model_selection import StratifiedKFold

parser = argparse.ArgumentParser()


parser.add_argument('-batch_size', '--batch_size', default=50, type=int)
parser.add_argument('--datapath', default=None, type=str)
parser.add_argument('--savepath', default=None, type=str)
parser.add_argument('--lr', default=3e-4, type=float)
parser.add_argument('--weight_decay', default=0.01, type=float)
parser.add_argument('--num_epochs', default=100, type=int)
parser.add_argument('--seed', default=1024, type=int)
parser.add_argument('--gpu', default="1", type=str)
parser.add_argument('--resume', default=False, type=bool)
parser.add_argument('--start_cross', default=0, type=int)
parser.add_argument('--meta', default=0, type=int)
parser.add_argument('--datasets', default=-1, type=int)
parser.add_argument('--mode', default="MLP", type=str)
parser.add_argument('--guidance', default="0,0", type=str)
parser.add_argument('--missing', default=False, type=bool)
parser.add_argument('--statistic', default=False, type=bool)
parser.add_argument('--testing', default=False, type=bool)
parser.add_argument('--accumulate', default=5, type=int)
parser.add_argument('--kfold', default=-1, type=int)
parser.add_argument('--data_prefix', default="/data/edu/", type=str, help="Prefix path for modality data files")
parser.add_argument('--model_path', default="/data/edu/PETCT/models/Longformer", type=str, help="Path to pretrained model")

path = os.path.dirname(__file__)
args = parser.parse_args()
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"


import torch
import torch.nn as nn
import torch.optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr

torch.autograd.set_detect_anomaly(True)
from model_semi import Model



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

    
def kf_main():
    s=1024
    models=["Trans"]
    initial_ckpts=args.savepath
    
    full_dataset = Fold_Dataset(root=args.datapath, meta=5, datasets= args.datasets, miss_mod= args.missing, data_prefix=args.data_prefix)
    
    if args.mode !="Multi":
        models = [args.mode]
        
    strat_keys = np.array([f"{m[-1]}_{int(r)}" for m, r in zip(full_dataset.files["Meta"], full_dataset.files["Result"])])
    indices = np.arange(len(strat_keys))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    outer_fold_results = []
    
    os.makedirs(initial_ckpts, exist_ok=True)
    
    
    # Set seeds
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    random.seed(s)
    np.random.seed(s)

    # Set cuda parameter
    cudnn.benchmark = False
    cudnn.deterministic = True

    def seed_worker(worker_id):
        worker_seed = s
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(s)
    
    for mod in models:
        os.makedirs(os.path.join(initial_ckpts,mod), exist_ok=True)
    
        for fold, (train_idx, test_idx) in enumerate(skf.split(indices, strat_keys)):
            
            if fold>=args.kfold:
            
                ckpts = os.path.join(initial_ckpts,mod,f"n_fold_{fold}")
                
                os.makedirs(ckpts, exist_ok=True)
        
                if args.resume==True:
                    log_file = open(os.path.join(ckpts, 'logging.txt'), 'a')
                else:
                    log_file = open(os.path.join(ckpts, 'logging.txt'), 'w+')

                
                start = time.time()
                print(f"\n{'='*20} ROUND {fold + 1}/5 {'='*20}")
                
                # --- PREPARE DATA FOR THIS ROUND ---
                # 80% Training
                full_dataset.set_indices(train_idx)
                full_dataset.val = 0 # Training mode (Augmentation ON)
                train_loader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=True, collate_fn = _custom_collate_fn,  num_workers=8, persistent_workers=True)
                
                test_dataset = Fold_Dataset(args.datapath, meta=args.meta, datasets=args.datasets, miss_mod=args.missing, data_prefix=args.data_prefix)
                test_dataset.set_indices(test_idx)
                test_dataset.val = 1 
                test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,collate_fn = _custom_collate_fn, num_workers=8, persistent_workers=True)
                
                modifications = [int(item) for item in args.guidance.split(',')]
                model = Model(args.meta, mod, full_dataset.max_lab(), full_dataset.task_types(), model_path=args.model_path).cuda().to(dtype=torch.bfloat16)
                
                if args.resume==True:
                    log_file = open(os.path.join(ckpts, 'logging.txt'), 'a')
                else:
                    log_file = open(os.path.join(ckpts, 'logging.txt'), 'w+')
                    
                log_file.write(str(args) + "\n")

                # --- CONSTRUCT THE OPT DICTIONARY ---
                opt = {
                    "mode": args.mode,
                    "resume": args.resume,
                    "tasks": full_dataset.obtain_tasks(),
                    "dist": full_dataset.get_dist(), 
                    "modifications": modifications, 
                    "path": os.path.join(ckpts),
                    "b_size": args.batch_size,
                    "weight_decay": args.weight_decay,
                    "lr": args.lr,
                    "TrainLoader": train_loader,
                    "ValLoader": test_loader 
                }

                training = Supervised_fold(model, args.num_epochs, opt,args)
                num_params = 0
                for param in model.parameters():
                    num_params += param.numel()
                print('Total number of parameters: %d' % num_params)
                training.train(args.num_epochs, full_dataset.max_lab())
                msg = 'total time: {:.4f} hours'.format((time.time() - start) / 3600)
                print(msg)
                log_file.write(msg + "\n\n\n")
                log_file.flush()
                
                if args.testing:
                    log_file = open(os.path.join(ckpts, 'logging_test.txt'), 'w+')
                    test = Testing(model,log_file,test_dataset,test_dataset.obtain_tasks(), s, ckpts)
                    test.test()
        
        
    
    
    

def main():
    seeds=[64,128,256,512,1024]
    models=[ "Roberta", "DBLora", "MLP", "Trans"]
    initial_ckpts=args.savepath
    train_set = Supervised_Dataset(root=args.datapath, val=0,  meta=5, datasets= args.datasets, miss_mod= args.missing, data_prefix=args.data_prefix)
    val_set = Supervised_Dataset(root=args.datapath,  val=1,  meta=5, datasets= args.datasets, miss_mod= args.missing, data_prefix=args.data_prefix)
    if args.testing == True:
        test_set = Supervised_Dataset(root=args.datapath,  val=2,  meta=5, datasets= args.datasets, miss_mod= args.missing, data_prefix=args.data_prefix)
        
    if args.mode !="Multi":
        models = [args.mode]
    for mod in models:
        model_ckpts = initial_ckpts+str(mod)+"/"
        if not args.statistic:
            seeds = [args.seed]
        for s in seeds:
            ckpts = model_ckpts+str(s)
            os.makedirs(ckpts, exist_ok=True)
            # Set seeds
            torch.manual_seed(s)
            torch.cuda.manual_seed(s)
            torch.cuda.manual_seed_all(s)
            random.seed(s)
            np.random.seed(s)

            # Set cuda parameter
            cudnn.benchmark = False
            cudnn.deterministic = True

            def seed_worker(worker_id):
                worker_seed = s
                np.random.seed(worker_seed)
                random.seed(worker_seed)

            g = torch.Generator()
            g.manual_seed(s)

            #Log files
            if args.resume==True:
                log_file = open(os.path.join(ckpts, 'logging.txt'), 'a')
            else:
                log_file = open(os.path.join(ckpts, 'logging.txt'), 'w+')

            start = time.time()
            msg = "Starting time: {:.1f}\n".format(start)
            log_file.write(msg)
            log_file.flush()
            log_file.write(str(args)+ "\n")
            log_file.flush()


            log_file = open(os.path.join(ckpts, 'logging.txt'), 'a')

            # Dataset
            train_loader = torch.utils.data.DataLoader(
                dataset=train_set,
                batch_size=args.batch_size,
                num_workers=10,
                shuffle=True,
                pin_memory=True,
                persistent_workers=True,
                worker_init_fn=seed_worker,
                generator=g,
                collate_fn = _custom_collate_fn
                )
            val_loader = torch.utils.data.DataLoader(
                dataset=val_set,
                batch_size=args.batch_size,
                num_workers=6,
                shuffle=True,
                pin_memory=False,
                worker_init_fn=seed_worker,
                generator=g,
                collate_fn = _custom_collate_fn)
            
                
            print(train_set.obtain_tasks())
            # Model + hyperparameters
            modifications = [int(item) for item in args.guidance.split(',')]
            model = Model(args.meta, mod, train_set.max_lab(), train_set.task_types(), model_path=args.model_path).cuda().to(dtype=torch.half)

            opt = {"mode":mod,"resume":args.resume,"tasks": train_set.obtain_tasks(),
                   "dist":train_set.dist, "modifications":modifications, "path": ckpts, "proportion": train_set.get_proportion(),  "b_size": args.batch_size,
                   "weight_decay": args.weight_decay, "lr": args.lr, "TrainLoader": train_loader, "ValLoader": val_loader}
            training = Supervised(model, args.num_epochs, opt,args)
            num_params = 0
            for param in model.parameters():
                num_params += param.numel()
            print('Total number of parameters: %d' % num_params)
            training.train(args.num_epochs, train_set.max_lab())
            msg = 'total time: {:.4f} hours'.format((time.time() - start) / 3600)
            print(msg)
            log_file.write(msg + "\n\n\n")
            log_file.flush()
            
            if args.testing:
                log_file = open(os.path.join(ckpts, 'logging_test.txt'), 'w+')
                test = Testing(model,log_file,test_set,train_set.obtain_tasks(), s, ckpts)
                test.test()
            
        


if __name__ == '__main__':
    if args.kfold!=-1:
        kf_main()
    else:   
        main()



