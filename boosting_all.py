from numpy import interp
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix, roc_curve,precision_recall_curve, auc
from sklearn.utils import shuffle
import csv
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pathlib
import xgboost as xgb
import sys

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)

method = "xgboost"
sim="Jaccard"
dataset="Dataset1"
sample="Random"

#数据集
pathlib.Path("../Dataset/"+dataset+"/Results/" + method+"_"+sim+"_"+sample).mkdir(parents=True, exist_ok=True)
sys.stdout = open("../Dataset/"+dataset+"/Results/" + method +"_"+sim +"_"+sample + "/" + str(method) + "Report.docx", "w")

def sen_and_spec(y_pred, y_real):
    tn, fp, fn, tp = confusion_matrix(y_real, y_pred).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    return sensitivity, specificity


output = []
benchmark = pd.read_csv("../Dataset/"+dataset+"/int_"+sample+"benchmark.csv", header=None)
benchmark = benchmark.values.tolist()
print("Method: " + method)
x_auc = []
y_auc = []

aucs = []

tprs=[]
tprs=[]
mean_fpr=np.linspace(0,1,100)

#维度确定为50维
for embed_size in range(10,156, 10):
    print("Embedding Size:", embed_size)
    df_embeddings = np.load("../Etra/" + dataset + "/output/embedding/" + sim + "_cmd/" + sim + "_cmd_" + str(embed_size) + ".npy")

    df_embeddings = pd.DataFrame(df_embeddings)
    df_embeddings['alle'] = [tuple(x) for x in
                             df_embeddings[[i for i in range(0, df_embeddings.shape[1])]].values.tolist()]
    emmbeddings = dict(df_embeddings.alle)
    # print(benchmark)
    x = []
    y = []
    for i, b in enumerate(benchmark):
        circ = emmbeddings[b[0]]
        dis = emmbeddings[b[1]]
        label = b[2]
        x.append([circ[circiter] for circiter in range(len(circ))] + [dis[disiter] for disiter in range(len(dis))])
        y.append(label)

    x, y = shuffle(x, y,random_state=7997)
    x = np.array(x)
    y = np.array(y)
    n = 5
    # x = normalize(x)  # fŭr normalization
    kfold = StratifiedKFold(n_splits=n, shuffle=True, random_state=799)

    stats = {"Accuracy": [], "Precision": [], "F1 score": [], "Sensitivity": [], "Specificity": [], "AUC": [],"aupr":[]}

    i = 0
    for train_ix, test_ix in kfold.split(x, y):
        print("----------------------------------------------------------------------------------------------")
        train, test = x[train_ix], x[test_ix]
        y_train, y_test = y[train_ix], y[test_ix]
        dtrain = xgb.DMatrix(train, label=y_train)
        param = {
            'eta': 0.3,#0.3
            'max_depth': 6,#6
            'objective': 'multi:softprob',
            'num_class': 2}
        dtest = xgb.DMatrix(test, label=y_test)
        evallist = [(dtrain, 'train')]
        num_round = 80
        bst = xgb.train(param, dtrain, num_round, evallist)
        y_pred = bst.predict(dtest, ntree_limit=bst.best_iteration+1)
        fpr, tpr, _ = roc_curve(y_test, y_pred[:, 1])
        roc_auc = auc(fpr, tpr)
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred[:, 1])
        # 平均
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        aucs.append(roc_auc)

        y_pred = np.asarray([np.argmax(line) for line in y_pred])
        sen, spec = sen_and_spec(y_pred, y_test)
        fold_stats = {"Fold Num": (i + 1)/100,
                      "Accuracy": accuracy_score(y_test, y_pred)/100,
                      "Precision": precision_score(y_test, y_pred)/100,
                      "Recall": recall_score(y_test, y_pred)/100,
                      "F1 score": f1_score(y_test, y_pred)/100,
                      "Sensitivity": sen/100,
                      "Specificity": spec/100,
                      "AUC": roc_auc/100,
                      "aupr":auc(recall, precision)/100
                      }
        fold_stats = {k: round(v*100, 3) for k, v in fold_stats.items()}
        fold_stats.pop("Recall", None)
        ordered_keys = ["Fold Num", "Accuracy", "F1 score", "Precision", "Sensitivity", "Specificity", "AUC","aupr"]
        fold_stats = {k: fold_stats[k] for k in ordered_keys}
        for k, v in fold_stats.items():
            print(k + ":", v)
        buffer = fold_stats.copy()
        buffer["Embedding"] = embed_size
        output.append(buffer)
        for k in stats:
            stats[k].append(fold_stats[k])
        print("----------------------------------------------------------------------------------------------")
        print()
        i += 1

    print("Overall:")
    for k, v in stats.items():
        print(k + ":", np.mean(v))
    buffer = {k: np.mean(v) for k, v in stats.items()}
    buffer["Embedding"] = embed_size
    buffer["Fold Num"] = 'average'
    output.append(buffer)
    o = pd.DataFrame(output)
    #o.to_excel("../Etra/MY/data3/" + method + "_" + sim +"_"+sample + "50.xlsx")
    #o.to_excel("../Etra/"+dataset+"/Results/" + method + "_" + sim +"_"+sample + "zhegnshu.xlsx")
    print()
    print("###############################################################################")
    print()

    x_auc.append(embed_size)
    y_auc.append(np.mean(stats["AUC"]))

#画不同特征的图for i_x, i_y in zip(x_auc, y_auc):

step_size = 10
plt.figure(figsize=(12, 6))
plt.plot(x_auc, y_auc, 'ro-', color='#0099CC',marker='o',markerfacecolor='red',markeredgecolor='red')
plt.title("XGboost")
for i_x, i_y in zip(x_auc, y_auc):
    plt.text(i_x, i_y, '{:.4f}'.format(i_y))
    # i_y=round(i_y, 3)
    # cm = plt.bar(i_x, i_y,width=4)
plt.xticks(np.arange(min(x_auc), max(x_auc) + 5, step_size))
plt.ylim(min(y_auc)-0.001,max(y_auc)+0.001)
plt.xlabel("characteristic dimension")
plt.ylabel("Mean AUC for 5 folds")


#跑所有维度的时候解除注销
#pathlib.Path("../Etra/"+dataset+"/Results/" + "FeatureSizes").mkdir(parents=True, exist_ok=True)
#with open("../Etra/"+dataset+"/Results/" + "FeatureSizes/" + method+"_"+sim +"_"+sample + ".csv", "w") as the_file:
#    csv.register_dialect("custom", delimiter=" ", skipinitialspace=True)
#    writer = csv.writer(the_file, dialect="custom")
#    for tup in list(zip(x_auc, y_auc)):
#        writer.writerow(tup)
#pathlib.Path("../Etra/"+dataset+"/Results/" + method +"_"+sim+"_"+sample + "/" + "FeatureSizes").mkdir(parents=True, exist_ok=True)
#plt.savefig("../Etra/"+dataset+"/Results/" + method +"_"+sim+"_"+sample + "/" + "FeatureSizes/" + method + "Feature_Comparisong改进折现" + "zhengshu.png")

sys.stdout.close()
