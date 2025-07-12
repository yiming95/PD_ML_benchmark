from torch.utils.data import Dataset
import torch
import numpy as np
import cv2
import random
import json
from torchvision import transforms
import os
import mne
import copy
import logging
import random

# 设置日志级别
mne.set_log_level('ERROR')

class Standardize:
    def __call__(self, sample):
        mean = sample.mean()
        std = sample.std()
        return (sample - mean) / std


class EEGDataset(Dataset):
    def __init__(self, root_dir, chtype, datasettype,transform=Standardize()):
        random.seed(42)  # 使用42作为随机种子
        self.files = os.listdir(root_dir)
        self.root_dir = root_dir
        self.datasetlist = {}
        self.transform = transform
        channel_names = [chtype]

        self.datalist = []
        self.labellist = []
        self.totalpieces = 0
        self.controlpieces = 0
        self.pdpieces = 0
        self.pds=0
        self.controls=0
        random.seed(42)
        for subject in self.files:
            if "sub" not in subject:
                continue
            for foldname in os.listdir(root_dir + subject):
                if ("-on" in foldname) and ("pd" in subject):
                    continue
                path = self.root_dir + subject + "/" + foldname + "/eeg/"
                if "pd" in subject:
                    self.pds+=1
                else:
                    self.controls+=1
                for file in os.listdir(path):
                    if ".bdf" in file:
                        self.datasetlist[file] = []

                        raw = mne.io.read_raw_bdf(path + file, preload=True)
                        picks = mne.pick_channels(raw.ch_names, include=channel_names)
                        cropped_data = raw.copy().crop(tmin=1.9)
                        data, times = cropped_data[picks, :]
                        counter = 0
                        while counter + 512 < data.shape[1]:
                            segment = data[:, counter:counter + 512][0]
                            if self.transform:
                                segment = self.transform(segment)
                            if "pd" in subject:
                                self.datasetlist[file].append((segment,1))
                                self.pdpieces+=1
                            else:
                                self.datasetlist[file].append((segment,0))
                                self.controlpieces+=1
                            counter += 512
                            self.totalpieces+=1
                        random.shuffle(self.datasetlist[file])

        print()
        random.seed(42)  # 使用42作为随机种子

        #划分方式可以更换
        if datasettype == "train":
            for keyname in self.datasetlist.keys():
                start = 0
                end = int(len(self.datasetlist[keyname]) * 0.8*0.8)
                for i in range(start,end):
                    self.datalist.append(self.datasetlist[keyname][i][0])
                    self.labellist.append(self.datasetlist[keyname][i][1])
        else:
            if datasettype == "valid":
                for keyname in self.datasetlist.keys():
                    start = int(len(self.datasetlist[keyname]) * 0.8 * 0.8)
                    end = int(start + len(self.datasetlist[keyname]) * 0.8 * 0.2)
                    for i in range(start, end):
                        self.datalist.append(self.datasetlist[keyname][i][0])
                        self.labellist.append(self.datasetlist[keyname][i][1])
            else:
                for keyname in self.datasetlist.keys():
                    start = int(len(self.datasetlist[keyname]) * 0.8 * 0.8 + len(self.datasetlist[keyname]) * 0.8 * 0.2)
                    end = int(len(self.datasetlist[keyname]))
                    for i in range(start, end):
                        self.datalist.append(self.datasetlist[keyname][i][0])
                        self.labellist.append(self.datasetlist[keyname][i][1])

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        inputs = self.datalist[idx]
        label = self.labellist[idx]
        inputs = torch.from_numpy(inputs).float() if not isinstance(inputs, torch.Tensor) else inputs
        label = torch.tensor(label)
        return inputs, label


# 现在实例化 EEGDataset 类时，加入 Standardize 转换
#datapath = "./UCSD/"
#inputdataset = EEGDataset(datapath, "P8", "train",transform=Standardize())
#print(inputdataset[0])

#datapath = "./UCSD/"
#inputdataset = EEGDataset(datapath,"P8")
#inputdataset[0]

"""
#inputdataset[0]
#inputdataset[1]
"""