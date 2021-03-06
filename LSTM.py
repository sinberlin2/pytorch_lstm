#!/usr/bin/env python
# coding: utf-8

#LSTM network river levels one station
import numpy as np # linear algebra
import os.path
from os import path  # accessing directory structure
import utils
import train_pytorch_lstm
import test_pytorch_lstm
from data_loader_river_data import DataLoader
import pytorch_lstm_model

#Define default model parameters
model_type= 'LSTM'
stateful_batches=0
stateful=1
hidden_layer_size=30  #30 stage
num_layers= 2
output_size= 1 #keep this 1, is the prediction size (1 day at a time is predicted)
epochs = 20
loss_metric = 'RMSE'
lr=0.002  #for stage prediction, 0.0002 was too low
lr_decay=0.0
fut_pred=10 # how many predictions are made into the future
batch_size=12
sliding_window=10
dropout=0.5 #droupout probability , NOT keep probability # we need dropout 0.5
load_model = False

#some good combo was dp 0.5 and hd30 and lr-0.002

base_path= 'C:/Users/doyle/Documents/Coding/HAL24K/'
sub_folder='darlaston'
data_folder= base_path + 'data/river_trent/' + sub_folder +'/'


### Define Model Inputs ###
input_size = 2 # no of features
# Define Prediction variable
pred_var = 'stage'
all_vars = ['stage', 'flow', 'rain']
stage = False
flow = False
rain = True
#specify which variables need to be scaled
scaler_vars = ['stage', 'flow', 'rain']

#add selected variables to dictionary
cond_vars_dict = dict(((k, eval(k)) for k in all_vars))

cond_vars_selected = {k: v for k, v in cond_vars_dict.items() if v is not False}
no_cond_vars=(sum(value == True for value in cond_vars_dict.values()))
assert no_cond_vars == input_size-1, "select the correct no of variables"

#set folder name for storing results

if input_size==1:
    input_names= '{}_unconditional'.format(pred_var)
else:
    input_names='__'.join(map(str, cond_vars_selected.keys()))
    input_names= '{}_{}'.format(pred_var, input_names)

results_folder= 'results_{}/'.format(model_type.lower())
if os.path.exists(base_path + results_folder)==False:
    os.mkdir(base_path + results_folder)

results_folder= results_folder + '{}/'.format(sub_folder)
if os.path.exists(base_path + results_folder)==False:
    os.mkdir(base_path + results_folder)

results_folder= results_folder + input_names +'/'
if os.path.exists(base_path + results_folder)==False:
    os.mkdir(base_path + results_folder)
    print('created results folder at', base_path + results_folder)
results_folder=base_path +results_folder


def main():
    print('cond lstm predicting ' + pred_var)
    data_loader= DataLoader(pred_var, sliding_window, output_size, input_size, base_path, data_folder, sub_folder, cond_vars_dict, scaler_vars)
    dataset=data_loader.load_data()
    train_inout_seq, val_inout_seq, test_inout_seq  = data_loader.split_scale_transform(dataset)

    if load_model == False:
            model=train_pytorch_lstm.train(train_inout_seq, val_inout_seq, sliding_window, epochs, batch_size, input_size, hidden_layer_size, num_layers, output_size, lr, lr_decay, dropout, results_folder, stateful, stateful_batches)
    else:
        model = pytorch_lstm_model.LSTM(batch_size=batch_size, input_size=input_size, seq_len=sliding_window,
                                        hidden_layer_size=hidden_layer_size,
                                        num_layers=num_layers, output_size=output_size,
                                        dropout=dropout)

        save_model = False
        if save_model is not False:
            model_path =  results_folder + 'LSTM_' + 'tw_' + str(sliding_window) + '.pt'
            model.load_state_dict(torch.load(model_path))

    #Evaluation of the model/ Testing
    yhat, u2_values, test_x, test_y =test_pytorch_lstm.test_model(model, test_inout_seq, input_size, fut_pred, sliding_window, stateful, stateful_batches)

    #Scale back data, inverse scaler expects array of shape  [samples, timesteps, features]
    test_x = data_loader.scale_back(test_x)
    test_preds = data_loader.scale_back(yhat)
    test_y = data_loader.scale_back(test_y)

    #plot all predictions of first index
    utils.plot_test_predictions(test_y, sliding_window, results_folder, u2_values, test_preds, pred_var, pred_index=0, show_last=False)
    # #plot last X predictions of first and last index
    utils.plot_test_predictions(test_y, sliding_window, results_folder, u2_values, test_preds, pred_var, pred_index=0, show_last=100)
    utils.plot_test_predictions(test_y, sliding_window, results_folder, u2_values, test_preds, pred_var, pred_index= -1, show_last=20)

    #for some random test data, plot all of its predictions
    for pred_no in [0, 30, 50, 144]:
        assert pred_no <= test_y.shape[0] - fut_pred, "make sure that prediction is within valid range"
        utils.plot_sample_prediction(test_x, test_y, test_preds, pred_no, sliding_window, fut_pred, output_size, results_folder, pred_var)


if __name__ == '__main__':
    main()
