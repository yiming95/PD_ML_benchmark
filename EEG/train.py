import argparse
import os.path as osp
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from EEGDataset import EEGDataset
from convnet import ANN
from utils import pprint, set_gpu, ensure_path, Averager, Timer, count_acc, euclidean_metric
from tqdm import tqdm
import numpy as np
import os
import mne
import json
import random
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader, random_split
from torch.optim.optimizer import Optimizer, required
import torch.nn as nn


# 多数投票组合
def majority_vote(models, test_loaders):
    all_predictions = []
    for i in range(len(models)):
        model = models[i]
        model.eval()
        model_predictions = []
        temp = test_loaders[i]
        for batch in temp:
            data, gt = [_ for _ in batch]
            pred = model(data)
            preds = (pred >= 0.5).cpu().numpy()
            model_predictions.append(preds)

        all_predictions.append(np.concatenate(model_predictions))

    all_predictions = np.array(all_predictions)
    majority_vote_predictions = np.round(np.mean(all_predictions, axis=0))

    return majority_vote_predictions



class PolakRibiereOptimizer(Optimizer):
    def __init__(self, params, lr=required, beta=0.1):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, beta=beta)
        super(PolakRibiereOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Polak-Ribiere does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['old_grad'] = torch.zeros_like(p.data)
                    state['d'] = -grad

                old_grad = state['old_grad']
                d = state['d']
                beta = group['beta']
                lr = group['lr']

                # Calculate Polak-Ribière beta factor
                # Using torch.sum and element-wise multiplication to compute the dot product for potentially multi-dimensional tensors
                beta_factor = (torch.sum((grad - old_grad) * grad) / torch.sum(old_grad * old_grad)) * beta
                d = -grad + beta_factor * d

                # Update parameters
                p.data.add_(d, alpha=lr)

                # Update old gradient and direction
                state['old_grad'] = grad
                state['d'] = d
                state['step'] += 1

        return loss

mne.set_log_level('WARNING')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(42)
criterion = nn.BCEWithLogitsLoss()

def rereangeshot(data,label):
    data = data.cpu().numpy()
    label = label.cpu().numpy()
    controlind = np.argwhere(label == 0).reshape(-1)
    pdind = np.argwhere(label == 1).reshape(-1)
    if len(controlind)<len(pdind):
        minlen = len(controlind)
    else:
        minlen = len(pdind)
    np.random.seed(42)
    selectcontrol=np.random.choice(controlind, size=minlen, replace=False)
    selectpd = np.random.choice(pdind, size=minlen, replace=False)

    controlselected_elements = np.array([data[i] for i in selectcontrol])
    pdselected_elements = np.array([data[i] for i in selectpd])
    result = torch.tensor(np.concatenate((controlselected_elements, pdselected_elements)))
    return result

def binary_accuracy(outputs, labels):
    """
    计算二分类问题的准确率。

    参数:
        outputs (torch.Tensor): 模型输出的概率，形状为 (N, 1) 或 (N,)
        labels (torch.Tensor): 真实的标签，形状为 (N,)

    返回:
        accuracy (float): 计算得到的准确率。
    """
    # 阈值处理，将概率值转换为0或1
    preds = outputs >= 0.5  # 输出大于等于0.5视为类别1
    # 确保labels是布尔型或者同样的0、1
    labels = labels.bool()

    # 计算正确预测的数量
    correct = torch.eq(preds, labels).sum().item()
    # 计算准确率
    accuracy = correct

    return accuracy

def specificity_sensitivity_accuracy(preds, labels):
    """
    计算二分类问题的特异性、敏感性和准确性。

    参数:
        preds (torch.Tensor): 预测标签，形状为 (N,)
        labels (torch.Tensor): 真实标签，形状为 (N,)

    返回:
        specificity (float): 计算得到的特异性。
        sensitivity (float): 计算得到的敏感性。
        accuracy (float): 计算得到的准确性。
    """
    try:
        cm = confusion_matrix(labels.cpu().numpy(), preds.cpu().numpy())
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        accuracy = (tn + tp) / (tn + fp + fn + tp)
    except ValueError as e:
        print(f"Error in confusion_matrix: {e}")
        print(f"labels shape: {labels.shape}, preds shape: {preds.shape}")
        specificity, sensitivity, accuracy = 0, 0, 0
    return specificity, sensitivity, accuracy

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.best_acc = None
        self.early_stop = False

    def __call__(self, val_acc, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
            self.best_acc = val_acc
        elif (val_acc < self.best_acc - self.min_delta) and (val_loss > self.best_score - self.min_delta):
            if epoch > 30:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
        else:
            if val_acc > self.best_acc:
                self.best_acc = val_acc
            if val_loss < self.best_score:
                self.best_score = val_loss
            self.counter = 0

def save_model(name):
    torch.save(model.state_dict(), osp.join(args.save_path, name + '.pth'))

def datasetsplit(datapath, fold, k_folds):
    pdsgroup = []
    controlsgroup = []

    for subject in os.listdir(datapath):
        if "sub" not in subject:
            continue
        if "pd" in subject:
            pdsgroup.append(subject)
        else:
            controlsgroup.append(subject)
    subjectsgroup = pdsgroup + controlsgroup

    random.shuffle(pdsgroup)
    random.shuffle(controlsgroup)
    ratio = 1 / k_folds
    if fold != k_folds - 1:
        vallist = (pdsgroup[int(round(fold * ratio * len(pdsgroup))):int(round((fold + 1) * ratio * len(pdsgroup)))] +
                   controlsgroup[int(round(fold * ratio * len(controlsgroup))):int(round((fold + 1) * ratio * len(controlsgroup)))])
    else:
        vallist = (pdsgroup[int(round(fold * ratio * len(pdsgroup))):] +
                   controlsgroup[int(round(fold * ratio * len(controlsgroup))):])

    trainlist = []
    for item in subjectsgroup:
        if item not in vallist:
            trainlist.append(item)
    return trainlist, vallist

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epoch', type=int, default=20)
    parser.add_argument('--save-epoch', type=int, default=20)
    parser.add_argument('--save-path', default='./output')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    pprint(vars(args))

    set_gpu(args.gpu)
    #ensure_path(args.save_path)

    trlog = {}
    trlog['args'] = vars(args)
    trlog['train_loss'] = {}
    trlog['val_loss'] = {}
    trlog['test_loss'] = {}
    trlog['train_acc'] = {}
    trlog['val_acc'] = {}
    trlog['test_acc'] = {}
    trlog['train_spe'] = {}
    trlog['val_spe'] = {}
    trlog['test_spe'] = {}
    trlog['train_sen'] = {}
    trlog['val_sen'] = {}
    trlog['test_sen'] = {}
    trlog['test_auc'] = {}

    #random.shuffle(subjectsgroup)
    previousch = "Oz"
    if os.path.exists("./output/trlog.json"):
        with open("./output/trlog.json", 'r') as file:
            trlog = json.load(file)
            previousch = len(trlog['train_loss'].keys())-1
            trlog['train_loss'][str(previousch)] = []
            trlog['val_loss'][str(previousch)] = []
            trlog['test_loss'][str(previousch)] = []
            trlog['train_acc'][str(previousch)] = []
            trlog['val_acc'][str(previousch)] = []
            trlog['test_acc'][str(previousch)] = []
            trlog['train_spe'][str(previousch)] = []
            trlog['val_spe'][str(previousch)] = []
            trlog['test_spe'][str(previousch)] = []
            trlog['train_sen'][str(previousch)] = []
            trlog['val_sen'][str(previousch)] = []
            trlog['test_sen'][str(previousch)] = []
            print()

    flag = False
    channels = ["Oz", "P8", "FC2"]
    best_models = {}
    for chname in channels:
        if chname == previousch:
            flag = True
        if not flag:
            continue
        datapath = "./UCSD/"

        # ---------------------------分割数据集-----------------------------------
        train_dataset = EEGDataset(datapath, chname, "train")
        valid_dataset = EEGDataset(datapath, chname, "valid")
        test_dataset = EEGDataset(datapath, chname, "test")
        batch_size = 64  # 或者根据你的模型和GPU内存调整

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # ---------------------------进行训练-----------------------------------
        print(f'Channel Name: {chname}')
        print('--------------------------------')
        args.save_path = './output/' + chname + '/'
        ensure_path(args.save_path)

        # 设置模型参数
        model = ANN()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # PolakRibiereOptimizer(model.parameters(), lr=0.01, beta=0.1)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        early_stopping = EarlyStopping(patience=20, min_delta=0.01)

        trlog['train_loss'][str(chname)] = []
        trlog['val_loss'][str(chname)] = []
        trlog['test_loss'][str(chname)] = []
        trlog['train_acc'][str(chname)] = []
        trlog['val_acc'][str(chname)] = []
        trlog['test_acc'][str(chname)] = []
        trlog['train_spe'][str(chname)] = []
        trlog['val_spe'][str(chname)] = []
        trlog['test_spe'][str(chname)] = []
        trlog['train_sen'][str(chname)] = []
        trlog['val_sen'][str(chname)] = []
        trlog['test_sen'][str(chname)] = []
        trlog['test_auc'][str(chname)] = []

        best_val_acc = 0
        best_val_loss = float('inf')
        timer = Timer()

        for epoch in range(1, args.max_epoch + 1):
            model.train()
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.max_epoch} - Training", leave=True)
            train_preds = []
            train_labels = []
            loss = None
            for batch in progress_bar:
                data, gt = [_ for _ in batch]
                gt = gt.unsqueeze(1)
                pred = model(data)
                loss = criterion(pred, gt.float())
                train_preds.append((pred >= 0.5).cpu())
                train_labels.append(gt.cpu())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_specificity, train_sensitivity, train_accuracy = specificity_sensitivity_accuracy(torch.cat(train_preds), torch.cat(train_labels))
                progress_bar.set_postfix(train_loss=loss.item(), trainaccuracy=f"{train_accuracy:.4f}")

            train_preds = torch.cat(train_preds)
            train_labels = torch.cat(train_labels)
            train_specificity, train_sensitivity, train_accuracy = specificity_sensitivity_accuracy(train_preds, train_labels)

            model.eval()

            progress_bar2 = tqdm(valid_loader, desc=f"Epoch {epoch}/{args.max_epoch} - Validation", leave=True)
            valid_preds = []
            valid_labels = []
            valid_loss = None
            for batch in progress_bar2:
                valid_data, valid_gt = [_ for _ in batch]
                valid_gt = valid_gt.unsqueeze(1)
                valid_pred = model(valid_data)
                valid_loss = criterion(valid_pred, valid_gt.float())

                valid_preds.append((valid_pred >= 0.5).cpu())
                valid_labels.append(valid_gt.cpu())

                valid_specificity, valid_sensitivity, valid_accuracy = specificity_sensitivity_accuracy(torch.cat(valid_preds), torch.cat(valid_labels))

                progress_bar2.set_postfix(valid_loss=valid_loss.item(), validaccuracy=f"{valid_accuracy:.4f}")

            valid_preds = torch.cat(valid_preds)
            valid_labels = torch.cat(valid_labels)
            valid_specificity, valid_sensitivity, valid_accuracy = specificity_sensitivity_accuracy(valid_preds, valid_labels)

            # 保存最佳模型
            if valid_accuracy > best_val_acc:
                best_val_acc = valid_accuracy
                save_model('best_acc')

            if valid_loss.item() < best_val_loss:
                best_val_loss = valid_loss.item()
                save_model('best_loss')

            trlog['train_loss'][str(chname)].append(loss.item())
            trlog['train_acc'][str(chname)].append(train_accuracy)
            trlog['val_loss'][str(chname)].append(valid_loss.item())
            trlog['val_acc'][str(chname)].append(valid_accuracy)
            trlog['train_spe'][str(chname)].append(train_specificity)
            trlog['val_spe'][str(chname)].append(valid_specificity)
            trlog['train_sen'][str(chname)].append(train_sensitivity)
            trlog['val_sen'][str(chname)].append(valid_sensitivity)

            with open('./output/trlog.json', 'w') as file:
                json.dump(trlog, file, indent=4)

            lr_scheduler.step()

            early_stopping(valid_accuracy, valid_loss.item())
            if early_stopping.early_stop:
                print("Early stopping")
                break

        # 测试模型
        model.eval()
        test_accuracy = 0
        totalitemcounter = 0
        test_predictlabel = np.array([])
        test_truelabel = np.array([])
        test_probabilities = np.array([])

        progress_bar3 = tqdm(test_loader, desc="测试", leave=True)
        for batch in progress_bar3:
            test_data, test_gt = [_ for _ in batch]
            test_gt = test_gt.unsqueeze(1)
            test_pred = model(test_data)
            totalitemcounter += test_data.shape[0]

            test_loss = criterion(test_pred, test_gt.float())
            accuracy = binary_accuracy(test_pred, test_gt)
            test_accuracy += accuracy

            preds = (test_pred >= 0.5).cpu().numpy()
            test_predictlabel = np.append(test_predictlabel, preds)
            test_truelabel = np.append(test_truelabel, test_gt.cpu().numpy())
            test_probabilities = np.append(test_probabilities, test_pred.detach().numpy())

            progress_bar3.set_postfix(test_loss=test_loss.item(), testaccuracy=f"{test_accuracy / totalitemcounter:.4f}")

        # 计算额外的评估指标
        cm = confusion_matrix(test_truelabel, test_predictlabel)
        tn, fp, fn, tp = cm.ravel()
        test_spe = tn / (tn + fp) if (tn + fp) > 0 else 0
        test_sen = tp / (tp + fn) if (tp + fn) > 0 else 0
        test_auc = roc_auc_score(test_truelabel, test_probabilities)

        # 记录测试结果
        trlog['test_loss'][str(chname)].append(test_loss.item())
        trlog['test_acc'][str(chname)].append(test_accuracy / totalitemcounter)
        trlog['test_spe'][str(chname)].append(test_spe)
        trlog['test_sen'][str(chname)].append(test_sen)
        trlog['test_auc'][str(chname)].append(test_auc)

        with open('./output/trlog.json', 'w') as file:
            json.dump(trlog, file, indent=4)

        # 保存测试后的模型
        save_model('final')

        best_models[chname] = model


    best_models = {}
    channels = ["Oz", "P8", "FC2"]
    for folder in os.listdir("./output/"):
        if "json" in folder:
            continue
        model = ANN()
        model.load_state_dict(torch.load("./output/" + folder + "/final.pth"))
        best_models[folder] = model



    batch_size=64
    # 载入测试数据
    test_dataset = {}
    test_loader = {}
    for ch in channels:
        test_dataset[ch] = EEGDataset("./UCSD/", ch, "test")  # 使用其中一个通道加载测试集
        test_loader[ch] = DataLoader(test_dataset[ch], batch_size=batch_size, shuffle=False)

    # 多数投票预测
    majority_vote_preds = majority_vote([best_models[ch] for ch in channels], [test_loader[ch] for ch in channels])

    # 计算最终评估指标
    test_truelabel = np.array([label for _, label in test_dataset["Oz"]])
    cm = confusion_matrix(test_truelabel, majority_vote_preds)
    tn, fp, fn, tp = cm.ravel()
    final_test_spe = tn / (tn + fp) if (tn + fp) > 0 else 0
    final_test_sen = tp / (tp + fn) if (tp + fn) > 0 else 0
    final_test_acc = (tn + tp) / (tn +fn +tp + fp)
    final_test_auc = roc_auc_score(test_truelabel, majority_vote_preds)

    print(f"Final Test Accuracy: {final_test_acc:.4f}")
    print(f"Final Test Specificity: {final_test_spe:.4f}")
    print(f"Final Test Sensitivity: {final_test_sen:.4f}")
    print(f"Final Test AUC: {final_test_auc:.4f}")
