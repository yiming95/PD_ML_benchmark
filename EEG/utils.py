import os
import shutil
import time
import pprint

import torch
from sklearn.metrics import confusion_matrix,roc_auc_score
import numpy as np



def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)


def ensure_path(path):
    if os.path.exists(path):
        #if input('{} exists, remove? ([y]/n)'.format(path)) != 'n':
        shutil.rmtree(path)
        os.makedirs(path)
        pass
    else:
        os.makedirs(path)


class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


def count_acc(logits, label,probabilities,predictlabel,truelabel):
    pred = torch.argmax(logits, dim=1)

    prob = torch.softmax(logits, dim=1)[:, 1].cpu().detach().numpy()
    predictions = torch.argmax(logits, dim=1).cpu().detach().numpy()
    # 真实类别
    true_labels = label.cpu().numpy()

    cm = confusion_matrix(true_labels, predictions)

    # 检查混淆矩阵的形状并适当处理
    if cm.shape == (1, 1):
        # 只有一个类别的情况
        if true_labels[0] == 0:
            TN, FP, FN, TP = cm[0, 0], 0, 0, 0
        else:
            TN, FP, FN, TP =0, cm[0, 0], 0, 0
    else:
        # 两个类别的正常情况
        TN, FP, FN, TP = cm.ravel()


    sensitivity = TP / (TP + FN) if TP + FN > 0 else 0
    specificity = TN / (TN + FP) if TN + FP > 0 else 0
    curaccuracy = (TP + TN) / (TP + FP + TN + FN)

    # AUC计算前检查
    if len(np.unique(true_labels)) > 1:
        auc = roc_auc_score(true_labels, prob)
    else:
        auc = float('nan')  # 不适用或不能计算

    probabilities = np.append(probabilities, prob)
    predictlabel = np.append(predictlabel, predictions)
    truelabel = np.append(truelabel, true_labels)


    cm = confusion_matrix(truelabel, predictlabel)

    # 检查混淆矩阵的形状并适当处理
    if cm.shape == (1, 1):
        # 只有一个类别的情况
        if truelabel[0] == 0:
            tn, fp, fn, tp = cm[0, 0], 0, 0, 0
        else:
            tn, fp, fn, tp =0, cm[0, 0], 0, 0
    else:
        # 两个类别的正常情况
        tn, fp, fn, tp = cm.ravel()
    avgaccuracy = (tp + tn) / (tp + tn + fp + fn)

    return curaccuracy,avgaccuracy,probabilities,predictlabel,truelabel



def dot_metric(a, b):
    return torch.mm(a, b.t())


def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits


class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)

_utils_pp = pprint.PrettyPrinter()
def pprint(x):
    _utils_pp.pprint(x)


def l2_loss(pred, label):
    return ((pred - label)**2).sum() / len(pred) / 2