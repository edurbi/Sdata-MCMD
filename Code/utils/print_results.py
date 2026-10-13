import numpy as np
import re
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import os


parser = argparse.ArgumentParser()

parser.add_argument('--input', default=None, type=str)
parser.add_argument("--n", default = 1, type = int)
parser.add_argument("--save", default = None, type = str)
parser.add_argument("--out_dir", default = "/data/edu/PETCT/Code/utils/images", type = str, help="Directory to save output plots")
args = parser.parse_args()

save_folder = os.path.join(args.out_dir, args.save) if args.save else args.out_dir
if not os.path.isdir(save_folder):
    os.makedirs(save_folder, exist_ok=True)

file = open(args.input,"r")
results_val =[]
results_train =[]
data=[]
data_train=[]
for i in range(0,args.n):
    results_val.append({"epoch":[],"auc":[],"acc":[],"TP":[], "TN":[],"FP":[], "FN":[],"Loss":[]})
    results_train.append({"epoch":[],"auc":[],"acc":[],"TP":[], "TN":[],"FP":[], "FN":[],"Loss":[]})

for cross in range(0,args.n):
    stop =0
    while stop ==0:
        line = file.readline()
        if re.match(r"Val for.*", line):
            line = line.split(" ")
            results_val[cross]["epoch"].append(int(line[3].split("/")[0]))
            results_val[cross]["Loss"].append(float(line[6][0:-1]))
            results_val[cross]["auc"].append(float(line[11].split(",")[0]))
            results_val[cross]["acc"].append(float(line[9].split(",")[0]))
            results_val[cross]["TP"].append(int(line[13].split(",")[0]))
            results_val[cross]["FP"].append(int(line[15].split(",")[0]))
            results_val[cross]["FN"].append(int(line[17].split(",")[0]))
            results_val[cross]["TN"].append(int(line[18].split("e")[2][0:-1]))
        elif re.match(r"Train for.*", line):
            line = line.split(" ")
            results_train[cross]["epoch"].append(int(line[3].split("/")[0]))
            results_train[cross]["Loss"].append(float(line[6][0:-1]))
            results_train[cross]["auc"].append(float(line[11].split(",")[0]))
            results_train[cross]["acc"].append(float(line[9].split(",")[0]))
            results_train[cross]["TP"].append(float(line[13].split(",")[0]))
            results_train[cross]["FP"].append(float(line[15].split(",")[0]))
            results_train[cross]["FN"].append(float(line[17].split(",")[0]))
            results_train[cross]["TN"].append(float(line[18].split("e")[2][0:-1]))
        elif re.match(r"total time", line) or not line:
            stop = 1
max_ts = {"TP":[], "FP":[], "FN": []}
for cross in range(0,args.n):
    data.append( pd.DataFrame.from_dict(results_val[cross]))
    data_train.append( pd.DataFrame.from_dict(results_train[cross]))
for col in data[0]:
    fig,ax = plt.subplots(2)
    if col == "auc" or col == "acc":
        max=0
        max_train =0
    for cross in range(0,args.n):
        if col == "auc" or col == "acc":
            max+=np.max(data[cross][col])
            max_train+=np.max(data_train[cross][col])
            if col == "auc":
                position = np.argmax(data[cross][col])
                max_ts["TP"].append(data[cross]["TP"][position])
                max_ts["FP"].append(data[cross]["FP"][position])
                max_ts["FN"].append(data[cross]["FN"][position])

        ax[0].plot(data[cross]["epoch"],data[cross][col], label = "Num_val:"+str(cross))
        ax[1].plot(data_train[cross]["epoch"],data_train[cross][col], label = "Train:"+str(cross))

    if col == "auc" or col == "acc":
        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
        kw = dict(xycoords='data',textcoords="axes fraction",
            bbox=bbox_props, ha="right", va="top")
        ax[0].set_xlim([0,100])
        ax[0].set_ylim([0,1])
        ax[1].set_xlim([0,100])
        ax[1].set_ylim([0,1])
        ax[0].annotate("Mean best value:"+str(max/args.n), xy=(50, max/args.n), xytext=(0.9,0.9),clip_on=False, **kw)
        ax[1].annotate("Mean best value:"+str(max_train/args.n), xy=(50, max_train/args.n), xytext=(0.9,0.9),clip_on=False, **kw)


    ax[0].legend(loc="upper left")
    ax[1].legend(loc="upper left")

    plt.savefig(os.path.join(save_folder, 'graph_'+str(col)+'.png'),bbox_inches='tight',pad_inches=0)
    plt.close()

TP = sum([x for x in max_ts["TP"]])
print(TP)
FP =sum( [x for x in max_ts["FP"]])
FN = sum([x for x in max_ts["FN"]])
precision = (TP/(TP+FP))
recall = (TP/(TP+FN))
print("Precision: "+str(precision))
print("Recall: "+ str(recall))
print("F1: "+str(2*(precision*recall/(precision+recall))))
