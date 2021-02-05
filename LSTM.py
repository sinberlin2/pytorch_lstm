#!/usr/bin/env python
# coding: utf-8
#https://stackabuse.com/time-series-prediction-using-lstm-with-pytorch-in-python/

#LSTM network river levels one station
import numpy as np # linear algebra
import os.path
from os import path  # accessing directory structure
import utils
import train_pytorch_lstm
import test_pytorch_lstm
from data_loader_river_data import DataLoader
import pytorch_model
import pytorch_model_init_ev_batch
#import pytorch_model_cond

#Define default model parameters
model_type= 'LSTM'
stateful=0
init_batch=1
hidden_layer_size=30
num_layers= 2
output_size= 1 #keep this 1
epochs = 20
loss_metric = 'RMSE'
lr=0.0002
lr_decay=0.0
fut_pred=1 # how many predictions are made into the future
batch_size=64
sliding_window=10
dropout=1.0 #keep probability
predict_size=1
load_model = False

base_path= 'C:/Users/doyle/Documents/Coding/HAL24K/'
sub_folder='darlaston'
data_folder= base_path + 'data/river_trent/' + sub_folder +'/'


### Define Model Inputs ###
input_size = 2 # no of features
# Define Prediction variable
pred_var = 'stage'
all_vars = ['stage', 'flow', 'rain'] #add weekday
stage = False
flow=False
rain=True
#add selected variables to dictionary
cond_vars_dict = dict(((k, eval(k)) for k in all_vars))

cond_vars_selected = {k: v for k, v in cond_vars_dict.items() if v is not False}
no_cond_vars=(sum(value == True for value in cond_vars_dict.values()))
assert no_cond_vars == input_size-1, "select the correct no of variables"
#print(cond_vars_dict)

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
    data_loader= DataLoader(pred_var, sliding_window, predict_size, input_size, base_path, data_folder, sub_folder, cond_vars_dict)
    dataset=data_loader.load_data()
    train_inout_seq, val_inout_seq, test_inout_seq  = data_loader.split_scale_transform(dataset)

    if load_model == False:
            model=train_pytorch_lstm.train(train_inout_seq, val_inout_seq, sliding_window, epochs, batch_size, input_size, hidden_layer_size, num_layers, output_size, lr, lr_decay, dropout, results_folder, stateful, init_batch)

    else:
        if init_batch == True:
            print('batches initialised with zeros.')
            model = pytorch_model_init_ev_batch.LSTM(batch_size=batch_size, input_size=input_size, seq_len=sliding_window,
                                                     hidden_layer_size=hidden_layer_size,
                                                     num_layers=num_layers, output_size=output_size,
                                                     dropout=dropout)
        else:
            model = pytorch_model.LSTM(batch_size=batch_size, input_size=input_size, seq_len=sliding_window,  #cond
                                       hidden_layer_size=hidden_layer_size,
                                       num_layers=num_layers, output_size=output_size,
                                       dropout=dropout)

        model_path =  results_folder + 'LSTM_' + 'tw_' + str(sliding_window) + '.pt'
        model.load_state_dict(torch.load(model_path))

    #Evaluation of the model/ Testing
    yhat, naive_preds, u2_values =test_pytorch_lstm.test_model(model, test_inout_seq, fut_pred, sliding_window, stateful, init_batch)

    #Scale back data, inverse scaler expects array of shape  [samples, timesteps, features]
    test_x=  np.asarray([item[0].tolist() for item in test_inout_seq])
    test_y= np.asarray([item[1].tolist() for item in test_inout_seq])

    if input_size ==1:
        yhat= np.expand_dims(yhat, axis=2)
        naive_preds = np.expand_dims(naive_preds, axis=2)
        test_y = np.expand_dims(test_y, axis=2)
        test_x = np.expand_dims(test_x, axis=2)

    test_x = data_loader.scale_back(test_x)
    test_preds = data_loader.scale_back(yhat)
    naive_preds= data_loader.scale_back(naive_preds)
    test_y = data_loader.scale_back(test_y)

    utils.plot_test_predictions(dataset, sliding_window, results_folder, u2_values[0], test_preds, pred_var, fut_pred, pred_index=0, show_last=False)
    # #plot last 50 predictions of first and last index
    utils.plot_test_predictions(dataset, sliding_window, results_folder, u2_values[0], test_preds, pred_var, fut_pred, pred_index=0, show_last=10)
    utils.plot_test_predictions(dataset, sliding_window, results_folder, u2_values[0], test_preds, pred_var, fut_pred, pred_index=-1, show_last=20)

    #for some random test data, plot all of its predictions
    for pred_no in [0, 30, 50,144 ]:  # last number has to be < len(lest_inout_seq)-fut_pred
        utils.plot_sample_prediction(test_x, test_y, test_preds, pred_no, sliding_window, fut_pred, output_size, results_folder, pred_var)


if __name__ == '__main__':
    main()
