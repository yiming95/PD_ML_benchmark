# -*- coding: utf-8 -*-
"""
@author: Nguyen Duc Minh
"""

import numpy as np
np.random.seed(2)
import tensorflow as tf
tf.config.run_functions_eagerly(True)
# tf.enable_eager_execution()
from tensorflow.keras import layers
from tensorflow.keras import Input
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout, Flatten, Conv1D
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras import backend as K
import uuid
import glob
import os
import random
import sys
from tensorflow import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
sys.path.append('../src')
np.random.seed(2)
import os
from scipy import stats
import argparse
# fix random seed for reproducibility
np.random.seed(2)  # 2
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
import datetime
import json
from Data import Data
from Result import Results
from utils import *



def save_training_state(model, optimizer, epoch, lr, folder, filename='training_state.json'):
    model.save_weights(os.path.join(folder, 'model_weights.h5'))
    optimizer_state = optimizer.get_weights()
    state = {
        'optimizer_state': optimizer_state,
        'epoch': int(epoch),
        'lr': int(lr)
    }
    with open(os.path.join(folder, filename), 'w') as f:
        json.dump(state, f)


def load_training_state(folder, filename='training_state.json'):
    state_path = os.path.join(folder, filename)
    weights_path = os.path.join(folder, 'model_weights.h5')
    if os.path.exists(state_path) and os.path.exists(weights_path):
        with open(state_path, 'r') as f:
            state = json.load(f)
        return state, weights_path
    return None, None


# from src.data_utils2 import Datas
# from src.results import Results,Results_level

# from src.algo import multiple_cnn1D, multiple_cnn1D5_level
# from src.data_utils import Data

def train(model, datas, lr, log_filename, filename, folder):
    """

    :param model: Initial untrained model
    :param datas:  data object
    :param lr: learning rate
    :param log_filename: filename where the training results will be saved ( for each epoch)
    :param filename: file where the weights will be saved
    :return:  trained model
    """
    X_train = datas.X_train
    y_train = datas.y_train
    X_val = datas.X_val
    y_val = datas.y_val

    logger = CSVLogger(log_filename, separator=',', append=True)

    initial_epoch = 0
    state, weights_path = load_training_state(folder)
    if state:
        model.load_weights(weights_path)
        rms = optimizers.Nadam(lr=state['lr'])
        rms.set_weights(state['optimizer_state'])
        initial_epoch = state['epoch'] + 1
    else:
        rms = optimizers.Nadam(lr=lr)

    for i in (np.arange(1, 4) * 5):  # 10-20    1-10
        if i < initial_epoch:
            continue
        checkpointer = ModelCheckpoint(filepath=filename, monitor='val_accuracy', verbose=1, save_best_only=True,
                                       save_freq='epoch')
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=20, verbose=0, mode='auto')
        callbacks_list = [checkpointer, early_stopping, logger]

        # print("Y_train")
        # print(len(y_train))
        history = model.fit(np.split(X_train, X_train.shape[2], axis=2), \
                            # history  = model.fit(X_data,\
                            y_train, \
                            verbose=1, \
                            shuffle=True, \
                            epochs=100, \
                            batch_size=110, \
                            # validation_data=(X_val, y_val),\
                            validation_data=(np.split(X_val, X_val.shape[2], axis=2), y_val), \
                            callbacks=callbacks_list)

        model.load_weights(filename)
        lr = lr / 2
        rms = optimizers.Nadam(lr=lr)
        model.compile(loss='binary_crossentropy', optimizer=rms, metrics=['accuracy'])
        save_training_state(model, rms, i, lr, folder, filename='training_state.json')
        return model


def train_classifier(args):
    '''
    Function that performs the detection of Parkinson
    :param args: Input arguments
    :return:
    '''
    exp_name = args.exp_name  # 当前训练类型文件夹名

    start_fold = 0 # 获取当前训练到的folder数
    for file in os.listdir(args.output):
        if int(file.split("_")[2]) > start_fold:
            start_fold = int(file.split("_")[2])
    datas = Data(args.input_data, 1, 100, pk_level=False)
    for i in range(0, 10):
        lr = 0.001
        subfolder = os.path.join(args.output, exp_name + '_' + str(i))
        file_result_patients = os.path.join(subfolder, 'res_pat.csv')
        file_result_segments = os.path.join(subfolder, 'res_seg.csv')

        val_results = Results(file_result_segments, file_result_patients)

        datas.separate_fold(i)                                  #获取当前数据
        print("currnet epoch = ", i)
        if i<start_fold:
            continue
        model_file = os.path.join(subfolder, "model.json")

        if not os.path.exists(subfolder):
            os.makedirs(subfolder)



        model = multiple_transformer(datas.X_data.shape[2])     #建立模型
        model_json = model.to_json()
        with open(model_file, "w") as json_file:
            json_file.write(model_json)

        # print('fold', str(i))

        log_filename = os.path.join(subfolder, "training.csv")
        w_filename = os.path.join(subfolder, "weights.hdf5")

        if "weights.hdf5" in os.listdir(subfolder):
            model.load_weights(w_filename)

        model = train(model, datas, lr, log_filename, w_filename, subfolder)
        # print('Validation !!')
        val_results.validate_patient(model, datas.X_val, datas.y_val, datas.count_val)


if __name__ == '__main__':
    np.random.seed(2)
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", default='../data', type=str)
    # '
    parser.add_argument("--exp_name", default='train_classifier', type=str, help='train_classifier ; train_severity')
    parser.add_argument("--output", default='output', type=str)
    args = parser.parse_args(args=[])
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    if args.exp_name == 'train_classifier':
        train_classifier(args)
    # if args.exp_name == 'train_severity':
    # train_severity(args)

