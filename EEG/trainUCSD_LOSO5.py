import argparse
import os.path as osp
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from Meta_UCSD_EEGDataset import EEGDataset
from samplers import CategoriesSampler
from convnet import Convnet
from utils import pprint, set_gpu, ensure_path, Averager, Timer, count_acc, euclidean_metric
from tqdm import tqdm
import numpy as np
import os
import mne
import json
import random
from sklearn.metrics import confusion_matrix,roc_auc_score

mne.set_log_level('WARNING')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(42)
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



class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.best_acc = None
        self.early_stop = False

    def __call__(self, val_acc,val_loss):
        if self.best_score is None:
            self.best_score = val_loss
            self.best_acc = val_acc
        elif (val_acc < self.best_acc - self.min_delta) and (val_loss > self.best_score - self.min_delta):
            if epoch>30:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
        else:
            if val_acc > self.best_acc:
                self.best_acc = val_acc
            if val_loss < self.best_score:
                self.best_score = val_loss
            self.best_score = val_acc
            self.counter = 0

def save_model(name):
    torch.save(model.state_dict(), osp.join(args.save_path, name + '.pth'))

def datasetsplit(datapath,fold,k_folds):
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
    parser.add_argument('--max-epoch', type=int, default=200)
    parser.add_argument('--save-epoch', type=int, default=20)
    parser.add_argument('--shot', type=int, default=5)
    parser.add_argument('--query', type=int, default=5)
    parser.add_argument('--train-way', type=int, default=2)
    parser.add_argument('--test-way', type=int, default=2)
    parser.add_argument('--save-path', default='./UCSD/ON/LOSO_5/proto')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    pprint(vars(args))

    set_gpu(args.gpu)
    #ensure_path(args.save_path)
    k_folds = 15
    args.query = args.shot
    ratio = 1/k_folds

    trlog = {}
    trlog['args'] = vars(args)
    trlog['train_loss'] = {}
    trlog['val_loss'] = {}
    trlog['train_acc'] = {}
    trlog['val_acc'] = {}
    trlog['max_valid_acc'] = {}
    trlog['max_train_acc'] = {}
    trlog['max_valid_sen']= {}
    trlog['max_valid_spe']= {}
    trlog['max_valid_auc']= {}

    trlog['train_spe'] = {}
    trlog['val_spe'] = {}
    trlog['train_sen'] = {}
    trlog['val_sen'] = {}
    trlog['train_auc'] = {}
    trlog['val_auc'] = {}

    #random.shuffle(subjectsgroup)
    previousfold = 0
    if os.path.exists("./UCSD/ON/LOSO_5/trlog.json"):
        with open("./UCSD/ON/LOSO_5/trlog.json", 'r') as file:
            trlog = json.load(file)
            previousfold = len(trlog['train_loss'].keys())-1
            trlog['train_loss'][str(previousfold)] = []
            trlog['val_loss'][str(previousfold)] = []
            trlog['train_acc'][str(previousfold)] = []
            trlog['val_acc'][str(previousfold)] = []
            trlog['max_valid_acc'][str(previousfold)] = 0
            trlog['max_train_acc'][str(previousfold)] = 0
            trlog['max_valid_sen'][str(previousfold)] = 0
            trlog['max_valid_spe'][str(previousfold)] = 0
            trlog['max_valid_auc'][str(previousfold)] = 0

            trlog['train_spe'][str(previousfold)] = []
            trlog['val_spe'][str(previousfold)] = []
            trlog['train_sen'][str(previousfold)] = []
            trlog['val_sen'][str(previousfold)] = []
            trlog['train_auc'][str(previousfold)] = []
            trlog['val_auc'][str(previousfold)] = []
            print()


    for fold in range(k_folds):
        if fold < previousfold:
            continue
        datapath = "../data/UCSD/"
        trainlist, vallist = datasetsplit(datapath, fold, k_folds)

        print(f'FOLD {fold}')
        print('--------------------------------')
        args.save_path = './UCSD/ON/LOSO_5/proto-' + str(fold)
        ensure_path(args.save_path)



        # 设置训练集
        trainset = EEGDataset(datapath, trainlist,"train")
        train_sampler = CategoriesSampler(trainset.get_labels(), 20,args.train_way, args.shot + args.query)
        train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler)

        # 设置验证集
        valset = EEGDataset(datapath, vallist,"valid")
        val_loader = DataLoader(valset, batch_size=32, shuffle=False)
        val_trainloader = DataLoader(trainset, batch_size=100, shuffle=False)


        #设置模型参数
        model = Convnet().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        early_stopping = EarlyStopping(patience=20, min_delta=0.01)

        trlog['train_loss'][str(fold)] = []
        trlog['val_loss'][str(fold)] = []
        trlog['train_acc'][str(fold)] = []
        trlog['val_acc'][str(fold)] = []
        trlog['max_valid_acc'][str(fold)] = 0
        trlog['max_train_acc'][str(fold)] = 0
        trlog['max_valid_sen'][str(fold)] = 0
        trlog['max_valid_spe'][str(fold)] = 0
        trlog['max_valid_auc'][str(fold)] = 0

        trlog['train_spe'][str(fold)] = []
        trlog['val_spe'][str(fold)] = []
        trlog['train_sen'][str(fold)] = []
        trlog['val_sen'][str(fold)] = []
        trlog['train_auc'][str(fold)] = []
        trlog['val_auc'][str(fold)] = []
        timer = Timer()

        for epoch in range(1, args.max_epoch + 1):
            lr_scheduler.step()
            model.train()

            tl = Averager()
            predictlabel = np.array([])
            truelabel = np.array([])
            probabilities = np.array([])
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.max_epoch} - Training", leave=True)
            for batch in progress_bar:
                data, datalabel = [_.to(device) for _ in batch]
                p = args.shot * args.train_way
                data_shot, data_query = data[:p], data[p:]
                data_shot_label, data_query_label = datalabel[:p], datalabel[p:]
                proto = model(data_shot)
                proto = proto.reshape(args.shot, args.train_way, -1).mean(dim=0)

                label = data_query_label.type(torch.cuda.LongTensor)

                logits = euclidean_metric(model(data_query), proto)
                loss = F.cross_entropy(logits, label)
                #print('epoch {}, train {}/{}, loss={:.4f} acc={:.4f}'.format(epoch, i, len(train_loader), loss.item(), acc))

                acc,ta,probabilities,predictlabel,truelabel = count_acc(logits, label,probabilities,predictlabel,truelabel)
                tl.add(loss.item())

                progress_bar.set_postfix(currentloss=loss.item(), currentaccuracy=f"{acc:.4f}",avgloss=tl.item(), avgaccuracy=f"{ta:.4f}",LR=optimizer.param_groups[0]['lr'])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                proto = None;
                logits = None;
                loss = None

            tl = tl.item()
            tn, fp, fn, tp = confusion_matrix(truelabel, predictlabel).ravel()
            ta = (tp + tn) / (tp + tn + fp + fn)
            tspe = tn / (tn + fp)
            tsen = tp / (tp + fn)

            # AUC计算前检查
            if len(np.unique(truelabel)) > 1:
                tauc = roc_auc_score(truelabel, probabilities)
            else:
                tauc = float('nan')  # 不适用或不能计算

            model.eval()

            vl = Averager()
            val_predictlabel = np.array([])
            val_truelabel = np.array([])
            val_probabilities = np.array([])


            all_data = []
            all_labels = []

            # 遍历DataLoader来累积数据
            for data, labels in val_trainloader:
                all_data.append(data)
                all_labels.append(labels)

            # 使用torch.cat来合并所有的数据和标签
            all_data = torch.cat(all_data, dim=0)
            all_labels = torch.cat(all_labels, dim=0)

            rerangedata = rereangeshot(all_data, all_labels)

            val_data_shot = rerangedata.to(device)
            val_proto = model(val_data_shot)
            val_proto = val_proto.reshape(int(val_data_shot.shape[0]/args.test_way), args.test_way, -1).mean(dim=0)

            progress_bar2 = tqdm(val_loader, desc=f"Epoch {epoch}/{args.max_epoch} - Validation", leave=True)
            for val_data_query,val_label_query in progress_bar2:
                label = val_label_query
                label = label.type(torch.cuda.LongTensor)

                logits = euclidean_metric(model(val_data_query), val_proto)
                val_loss = F.cross_entropy(logits, label)

                vl.add(val_loss.item())
                val_acc, va, val_probabilities, val_predictlabel, val_truelabel = count_acc(logits, label, val_probabilities, val_predictlabel, val_truelabel)

                progress_bar2.set_postfix(currentloss=val_loss.item(), currentaccuracy=f"{val_acc:.4f}", avgloss=vl.item(),avgaccuracy=f"{va:.4f}")

                val_logits = None;
                val_loss = None


            vl = vl.item()
            tn, fp, fn, tp = confusion_matrix(val_truelabel, val_predictlabel).ravel()
            va = (tp + tn) / (tp + tn + fp + fn)
            vspe = tn / (tn + fp)
            vsen = tp / (tp + fn)
            #print('epoch {}, val, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))

            # AUC计算前检查
            if len(np.unique(truelabel)) > 1:
                vauc = roc_auc_score(val_truelabel, val_probabilities)
            else:
                vauc = float('nan')  # 不适用或不能计算


            if ((va > trlog['max_valid_acc'][str(fold)]) and (va<=ta)):
                trlog['max_valid_acc'][str(fold)] = va
                trlog['max_train_acc'][str(fold)] = ta
                trlog['max_valid_sen'][str(fold)] = vsen
                trlog['max_valid_spe'][str(fold)] = vspe
                trlog['max_valid_auc'][str(fold)] = vauc

                save_model('max-acc-vata')
            if va > trlog['max_valid_acc'][str(fold)]:
                save_model('max-acc')


            trlog['train_loss'][str(fold)].append(tl)
            trlog['train_acc'][str(fold)].append(ta)
            trlog['val_loss'][str(fold)].append(vl)
            trlog['val_acc'][str(fold)].append(va)
            trlog['train_spe'][str(fold)].append(tspe)
            trlog['val_spe'][str(fold)].append(vspe)
            trlog['train_sen'][str(fold)].append(tsen)
            trlog['val_sen'][str(fold)].append(vsen)
            trlog['train_auc'][str(fold)].append(tauc)
            trlog['val_auc'][str(fold)].append(vauc)


            with open('./UCSD/ON/LOSO_5/trlog.json', 'w') as file:
                json.dump(trlog, file, indent=4)
            with open('./UCSD/ON/LOSO_5/trlog.txt', 'w') as f:
                for key1, value2 in trlog.items():
                    f.write(key1)
                    f.write(":\n")
                    for key2, value2 in trlog[key1].items():
                        if key1 == 'args':
                            continue

                        if key1 == 'max_valid_acc' or key1 == 'max_train_acc' or key1 == 'max_valid_sen' or key1 == 'max_valid_spe' or key1 == 'max_valid_auc':
                            f.write(str(trlog[key1][key2]))
                            f.write("\n")
                        else:
                            f.write('fold-')
                            f.write(key2)
                            f.write(":     ")
                            for itemvalue in trlog[key1][key2]:
                                f.write(str(itemvalue))
                                f.write(" ")
                            f.write("\n")

            #torch.save(trlog, osp.join('./save', 'trlog.txt'))

            save_model('epoch-last')

            if epoch % args.save_epoch == 0:
                save_model('epoch-{}'.format(epoch))

            #print('ETA:{}/{}'.format(timer.measure(), timer.measure(epoch / args.max_epoch)))
            early_stopping(va,vl)
            if early_stopping.early_stop:
                print("Early stopping")
                break