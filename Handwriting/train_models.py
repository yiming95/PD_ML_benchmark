# coding=utf-8
import os
import sys
import pickle
import argparse
from pathlib import Path
from typing import List, Tuple
from datetime import datetime
sys.path.append('/')
import pandas as pd
import numpy as np
import sklearn
from keras import optimizers
from keras import callbacks
from keras import layers
from keras import models

from scripts import loader
from scripts import signals
from scripts import constants as c
from scripts.attention import AttentionLayer
from models import compute_metrics
import tensorflow as tf
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 这一行注释掉就是使用gpu，不注释就是使用cpu

N_EPOCHS = 30
RANDOM_SEED = 92
PREDICTIONS_FOLDER = Path('models/preds/50_50/')

def calculate_precision(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision = tp / (tp + fp)
    return precision

def calculate_recall(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    recall = tp / (tp + fn)
    return recall


def calculate_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    return specificity

def calculate_sensitivity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    return sensitivity

class MetricsLogger(Callback):
    def __init__(self, log_dir, validation_data):
        super(MetricsLogger, self).__init__()
        self.log_dir = log_dir
        self.validation_data = validation_data
        self.file_writer = tf.summary.create_file_writer(log_dir)

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        val_data, val_labels = self.validation_data
        y_pred_val = (self.model.predict(val_data) > 0.5).astype("int32")

        val_sensitivity = calculate_sensitivity(val_labels, y_pred_val)
        val_specificity = calculate_specificity(val_labels, y_pred_val)
        val_precision = calculate_precision(val_labels, y_pred_val)
        val_recall = calculate_recall(val_labels, y_pred_val)

        with self.file_writer.as_default():
            tf.summary.scalar('val_sensitivity', val_sensitivity, step=epoch)
            tf.summary.scalar('val_specificity', val_specificity, step=epoch)
            tf.summary.scalar('val_precision', val_precision, step=epoch)
            tf.summary.scalar('val_recall', val_recall, step=epoch)

def get_parameters() -> argparse.Namespace:
    parser = argparse.ArgumentParser(usage='Trains a m-GRU model')
    parser.add_argument('--drop', help='Dropout rate for recurrent and input layers', type=float,
                        required=True)
    parser.add_argument('--uni', help='Swap biGRU by GRU', action='store_true')
    parser.add_argument('--res', nargs='+', help='Sampling resultions. ' +
                        'Negative values generates samplings in the reverse order. ' + 
                        'Zero uses equally spaced samplings',
                        type=int, required=True)
    parser.add_argument('--lens', nargs='+', type=int, required=True)
    parser.add_argument('--ds', help='Dataset, either "sp" or "mea"', type=str, required=True)
    parser.add_argument('--ratio', help='Either 50_50 or 75_25', type=str, required=True)
    parser.add_argument('--hidden', help='Hidden layers size', type=int, default=64)
    parser.add_argument('--nl', help='Amount of recurrent layers', type=int, default=2)
    parser.add_argument('--n_models', help='Amount of models to be trained', type=int, default=10)
    args = parser.parse_args()
    
    print(f'Dataset:        {args.ds}')
    print(f'Split ratio:    {args.ratio}')
    print(f'Dropout rate:   {args.drop}')
    print(f'Bidirectional?  {not args.uni}')
    print(f'# RNN layers:   {args.nl}')
    print(f'# Rounds:       {args.n_models}')
    print(f'Hidden size:    {args.hidden}')
    print('Resolutions:    [{}]'.format(', '.join(map(str, args.res))))
    print('Lengths:        [{}]'.format(', '.join(map(str, args.lens))))
    sys.stdout.flush()
    return args


def build_model(hidden_sz: int,
                drop_prob: float,
                resolutions: List[int],
                lengths: List[int],
                n_recurrent_layers: int,
                bidirectional: bool) -> models.Model:
    def build_input(input_id, input_maxlen):
        signal_inp = layers.Input(shape=(input_maxlen, 6), name=f'res_{input_id}')
        signal_msk = layers.Masking(mask_value=0, name=f'mask-{input_id}')(signal_inp)
        signal_drp = layers.Dropout(rate=drop_prob, name=f'dropout-{input_id}')(signal_msk)
        return signal_inp, signal_drp

    def build_gru(units, name, bidirectional):
        rnn = layers.GRU(units=units, recurrent_dropout=drop_prob, return_sequences=True)
        if bidirectional:
            rnn = layers.Bidirectional(rnn, name=name)
        return rnn

    # Input part
    bottom_inputs = list()
    bottom_outputs = list()
    
    for input_id, (resolution, length) in enumerate(zip(resolutions, lengths)):
        # input part
        signal_in, signal_drop = build_input(resolution, length)
        
        # RNN part
        loop_input = signal_drop
        for i in range(1, n_recurrent_layers + 1):
            loop_input = build_gru(units=hidden_sz,
                                   name=f'gru-{input_id + 1}-{i}',
                                   bidirectional=bidirectional
                                  )(loop_input)
        x = loop_input

        # Descriptor part
        x = AttentionLayer(name=f'attention-{input_id + 1}')(x)
        
        bottom_inputs.append(signal_in)
        bottom_outputs.append(x)

    if len(resolutions) > 1:
        x = layers.concatenate(bottom_outputs, name='concat')
    else:
        x = bottom_outputs[0]

    x = layers.Dropout(drop_prob / 2, name='concat-drop')(x)
    x = layers.Dense(hidden_sz, activation='relu', name='middle')(x)
    x = layers.Dense(1, activation='sigmoid', name='output')(x)
    return models.Model(inputs=bottom_inputs, outputs=[x])


def basic_stats(rol: List[float]) -> Tuple[float, float, float, float]:
    return np.average(rol), np.std(rol), np.max(rol), np.min(rol)


def train_models(exam: str,
                 n_models: int,
                 dropout: float,
                 resolutions: List[int],
                 lengths: List[int],
                 n_recurrent_layers: int,
                 is_bidirectional: bool,
                 hidden_sz: int,
                 n_epochs: int,
                 ratio: str,
                 verbose: bool = True,
                 trn_fraction: float = 1) -> float:
    print(f'!!!!! Using RATIO: {ratio}')
    if exam == 'sp':
        print('loading spiral clips')
        if ratio == '75_25':
            statistics = c.spiral75_constants
        else:
            statistics = c.spiral50_constants
    elif exam == 'mea':
        print('loading meander clips')
        if ratio == '75_25':
            statistics = c.meander75_constants
        else:
            statistics = c.meander50_constants
    else:
        raise ValueError('--ds should be either "sp" or "mea"')

    x_trn, y_trn = loader.load_handp_v2(f'../data/_{exam}_{ratio}/', fold_id=0, sampling_factor=1, verbose=verbose)
    x_val, y_val = loader.load_handp_v2(f'../data/_{exam}_{ratio}/', fold_id=1, sampling_factor=1, verbose=verbose)
    x_tst, y_tst = loader.load_handp_v2(f'../data/_{exam}_{ratio}/', fold_id=2, sampling_factor=1, verbose=verbose)

    lens = [len(x) for x in x_trn]
    print('Mean: {} ± {}'.format(np.mean(lens), np.std(lens)))
    print('Max: {}'.format(np.max(lens)))
    print('Min: {}'.format(np.min(lens)))

    # Removing outliers
    x_trn = signals.clip_values(x_trn, statistics)
    x_val = signals.clip_values(x_val, statistics)
    x_tst = signals.clip_values(x_tst, statistics)

    # Shuffling. Notice that we use a *fixed* random seed for learning curves
    x_trn, y_trn = sklearn.utils.shuffle(x_trn, y_trn, random_state=RANDOM_SEED)

    train_cap = round(trn_fraction * len(x_trn))
    x_trn = x_trn[:train_cap]
    y_trn = y_trn[:train_cap]
    print(f'Training with {len(x_trn)} samples')

    # Computing statistics for normalization. It could be any other
    norm_stats = signals.compute_series_stats(x_trn, signals.mean_std)

    # The normalization
    x_trn = signals.normalize_multiseries(x_trn, norm_stats)
    x_val = signals.normalize_multiseries(x_val, norm_stats)
    x_tst = signals.normalize_multiseries(x_tst, norm_stats)

    # Sampling the signals. Samples ordering must be kept to use the previous labels
    bag_fts, bag_len = signals.multires_sampling(x_trn, x_val, x_tst, resolutions, lengths)

    model_names = list()
    all_acc_val = list()
    all_acc_tst = list()
    all_sens_val = list()
    all_sens_tst = list()
    all_spec_val = list()
    all_spec_tst = list()
    all_prec_val = list()
    all_prec_tst = list()
    all_recall_val = list()
    all_recall_tst = list()

    metrics_data = []

    for model_no in range(1, n_models + 1):
        print(f'============[Training model {model_no} / {n_models} ]============')

        # Building the model
        opt = optimizers.adam_v2.Adam(lr=5e-4, clipnorm=1.0)#optimizers.Adam(lr=5e-4, clipnorm=1.0)
        model = build_model(hidden_sz,
                            dropout,
                            resolutions,
                            lengths,
                            n_recurrent_layers,
                            is_bidirectional)
        model.compile(opt, loss='binary_crossentropy', metrics=['binary_accuracy'])
        model.summary()
        #-------windows setting--------------
        time_name = str(datetime.now())
        model_name = 'amodel_num_' +str(model_no)+"_"+ time_name.split(" ")[0] + "-" +time_name.split(" ")[1].split(".")[0].replace(':', '-')
        log_dir = f'./logs/50_50/{model_name}'
        cbs = [
            callbacks.ModelCheckpoint(f'trained_models/50_50/{model_name}',
                                      save_best_only=True,
                                      verbose=1),
            callbacks.TensorBoard(f'./logs/50_50/{model_name}'),
            MetricsLogger(log_dir=log_dir, validation_data=(bag_fts['val'], y_val))
        ]

        # Fitting
        model.fit(bag_fts['trn'], y_trn,
                  validation_data=(bag_fts['val'], y_val),
                  batch_size=16, epochs=n_epochs, callbacks=cbs)

        # Evaluating
        best_model = models.load_model(f'trained_models/50_50/{model_name}',
                                       {'AttentionLayer': AttentionLayer})


        y_pred_val = (best_model.predict(bag_fts['val']) > 0.5).astype("int32")
        y_pred_tst = (best_model.predict(bag_fts['tst']) > 0.5).astype("int32")

        _, acc_val = best_model.evaluate(bag_fts['val'], y_val)
        _, acc_tst = best_model.evaluate(bag_fts['tst'], y_tst)

        sens_val = calculate_sensitivity(y_val, y_pred_val)
        sens_tst = calculate_sensitivity(y_tst, y_pred_tst)
        spec_val = calculate_specificity(y_val, y_pred_val)
        spec_tst = calculate_specificity(y_tst, y_pred_tst)
        prec_val = calculate_precision(y_val, y_pred_val)
        prec_tst = calculate_precision(y_tst, y_pred_tst)
        recall_val = calculate_recall(y_val, y_pred_val)
        recall_tst = calculate_recall(y_tst, y_pred_tst)

        all_acc_val.append(acc_val)
        all_acc_tst.append(acc_tst)
        all_sens_val.append(sens_val)
        all_sens_tst.append(sens_tst)
        all_spec_val.append(spec_val)
        all_spec_tst.append(spec_tst)
        all_prec_val.append(prec_val)
        all_prec_tst.append(prec_tst)
        all_recall_val.append(recall_val)
        all_recall_tst.append(recall_tst)
        model_names.append(model_name)


        print('Val accuracy: {} {} {} {} (current: {})'.format(*basic_stats(all_acc_val), acc_val))
        print('Tst accuracy: {} {} {} {} (current: {})'.format(*basic_stats(all_acc_tst), acc_tst))
        print('Val sensitivity: {} {} {} {} (current: {})'.format(*basic_stats(all_sens_val), sens_val))
        print('Tst sensitivity: {} {} {} {} (current: {})'.format(*basic_stats(all_sens_tst), sens_tst))
        print('Val specificity: {} {} {} {} (current: {})'.format(*basic_stats(all_spec_val), spec_val))
        print('Tst specificity: {} {} {} {} (current: {})'.format(*basic_stats(all_spec_tst), spec_tst))
        print('Val precision: {} {} {} {} (current: {})'.format(*basic_stats(all_prec_val), prec_val))
        print('Tst precision: {} {} {} {} (current: {})'.format(*basic_stats(all_prec_tst), prec_tst))
        print('Val recall: {} {} {} {} (current: {})'.format(*basic_stats(all_recall_val), recall_val))
        print('Tst recall: {} {} {} {} (current: {})'.format(*basic_stats(all_recall_tst), recall_tst))

        metrics_data.append({
            'model_name': model_name,
            'val_accuracy': acc_val,
            'test_accuracy': acc_tst,
            'val_sensitivity': sens_val,
            'test_sensitivity': sens_tst,
            'val_specificity': spec_val,
            'test_specificity': spec_tst,
            'val_precision': prec_val,
            'test_precision': prec_tst,
            'val_recall': recall_val,
            'test_recall': recall_tst
        })

        # Persisting predictions, so we can compute wathever metric
        y_hat_val = best_model.predict(bag_fts['val'])
        y_hat_tst = best_model.predict(bag_fts['tst'])

        pickle.dump({
            'y_hat_val': y_pred_val,
            'y_hat_tst': y_pred_tst,
            'y_val': y_val,
            'y_tst': y_tst,
            'acc_val': acc_val,
            'acc_tst': acc_tst,
            'sens_val': sens_val,
            'sens_tst': sens_tst,
            'spec_val': spec_val,
            'spec_tst': spec_tst,
            'prec_val': prec_val,
            'prec_tst': prec_tst,
            'recall_val': recall_val,
            'recall_tst': recall_tst
        }, (PREDICTIONS_FOLDER / model_name).open('wb'))

        # Save metrics to CSV
        df = pd.DataFrame(metrics_data)
        df.to_csv("./csvresults/50_50/" + model_name + '.csv', index=False)

        del model

    print('Trained model names:')
    print(model_names)

    compute_metrics.print_report([os.path.join(PREDICTIONS_FOLDER, model_name)
                                  for model_name in model_names])

    val_acc_avg = basic_stats(all_acc_val)[0]

    return val_acc_avg


if __name__ == '__main__':
    params = get_parameters()
    train_models(params.ds, params.n_models, params.drop, params.res,
                 params.lens, params.nl, not params.uni, params.hidden,
                 N_EPOCHS, params.ratio)
