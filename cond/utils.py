import matplotlib.pyplot as plt # plotting
import numpy as np# linear algebra
import torch.nn as nn
import torch
import pandas as pd


def loss_function(x,y):
    criterion=nn.MSELoss()
    loss=torch.sqrt(criterion(x,y))
    return loss

def plot_loss(epochs, results_folder, train_losses, val_losses):
    x_grid = list(range(epochs))
    plt.title('Loss Across Epochs')
    plt.ylabel('RMSE Loss')
    plt.xlabel('Epochs')
    plt.xticks(x_grid)
    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='val')
    fig_path =  results_folder  + '_loss'
    plt.savefig(fig_path)
    # plt.show()
    plt.close('all')


def get_u2_value(y_pred, y_true, y_prev):  #x is y_pred

    y_true=np.squeeze(y_true)
    y_pred=np.squeeze(y_pred)
    y_prev=np.squeeze(y_prev)

    nom1 =np.subtract(y_pred, y_true)
    nom1 =np.square(nom1)
    nom2= np.square(y_prev)

    nom = np.divide(nom1 ,nom2, out=np.zeros_like(nom1), where=nom2!=0)
    nom = nom.mean()
    nom = np.sqrt(nom)

    denom1= np.subtract(y_true, y_prev)
    denom1 = np.square(denom1)
    denom2= np.square(y_prev)
    denom = np.divide(denom1,denom2,out=np.zeros_like(denom1), where=denom2!=0)

    denom= denom.mean()
    denom = np.sqrt(denom)

    u2_value= nom/denom
    return u2_value



#could add option to only plot the last values
def plot_test_predictions(dataset, tw, results_folder, u2_value,  test_preds, pred_var, fut_pred, pred_index=0, show_last= 50):

    #get the relevant test data (first prediction day etc). By default this is the first prediction day
    test_preds_ind = [item[pred_index] for item in test_preds]
    test_preds_ind = np.squeeze(test_preds_ind)

    # plot the last test  data
    #get the start of the test data including the first train window
    test_x_len= test_preds.shape[0]

    if show_last is not False:
        test_x_len = show_last
        test_preds_ind = test_preds_ind[-show_last:]

    test_data = dataset[-test_x_len:]
    #plot test  data
    x = np.arange(0, test_x_len, 1)
    plt.plot(x, test_data, label='test data')
    #print(x)

    # #plot predictions
    #get the delay for plotting the predictions
    if pred_index < 0:
        pred_delay = fut_pred  + pred_index +1  #maybe adjust this with multiple predictions
    else:
        pred_delay =  pred_index +1

    x = np.arange(pred_delay , pred_delay + test_preds_ind.shape[0], 1)
    #print(x)
    plt.plot(x, test_preds_ind, label='predictions')

    #crate plot
    plt.suptitle('LSTM Predictions of ' +pred_var + ' for day ' + str(pred_index), fontsize=12)
    plt.title('u2 value: '+ str(u2_value) , fontsize=10)
    plt.ylabel(pred_var)
    plt.xlabel('Days')
    plt.grid(False)
    plt.autoscale(axis='x', tight=True)
    plt.legend(loc='upper left', frameon=False)
    fig_path = results_folder + 'Test_predictions' + str(pred_index) + '_ft_' + str(fut_pred)
    plt.savefig(fig_path)
    #plt.show()
    plt.close('all')



def plot_sample_prediction(test_x, test_y, test_preds, pred_no, tw, fut_pred, output_size, results_folder, pred_var):
    # original test data
    test_input = test_x[pred_no]
    test_actual_data = test_y[pred_no: pred_no + fut_pred *output_size].flatten()   #maybe get rid of times outputsize
    test_pred = test_preds[pred_no]

    plt.title('LSTM Prediction No: ' + str(pred_no))
    plt.ylabel(pred_var)
    plt.xlabel('Days')
    plt.grid(False)
    plt.autoscale(axis='x', tight=True)

    # plot relevant portion of actual data
    plt.plot(test_input, label='input test data')

    # plot actual data for predction window
    x = np.arange(tw, tw + fut_pred*output_size, 1)
    #print(x.shape)
    plt.plot(x, test_actual_data, label='actual data')
    #plot predictions
    x = np.arange(tw, tw + fut_pred*output_size, 1)
    plt.plot(x, test_pred, label='prediction',color='teal',linestyle='--')
    plt.legend(loc='upper left', frameon=False)

    file_name = 'Test_prediction_no' + str(pred_no)
    plt.savefig(results_folder + file_name)
    #plt.show()
    plt.close('all')





