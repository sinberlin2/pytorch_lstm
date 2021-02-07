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
    plt.legend()
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


def plot_test_predictions( test_y, tw, results_folder, u2_value,  test_preds, pred_var, pred_index=0, show_last= 50):

    #get the relevant test data (first prediction day etc). By default this is the first prediction day
    #there is no delay between labels and predictions
    test_preds_sel=test_preds[:,pred_index, 0]
    test_y_sel=test_y[:,pred_index, 0]

    # plot the last test  data
    #get the start of the test data including the first train window
    test_x_len= test_preds.shape[0]

    if show_last is not False:
        test_x_len = show_last
        test_preds_sel = test_preds_sel[-show_last:]
        test_y_sel = test_y_sel[-show_last:]

    #plot test  data
    x = np.arange(0, test_x_len, 1)
    plt.plot(x, test_y_sel, label='test data')

    # #plot predictions
    plt.plot(x, test_preds_sel, label='predictions')

    #create plot
    plt.suptitle('LSTM Predictions of ' +pred_var + ' for a ' + str(pred_index + 1) + 'day prediction', fontsize=12)
    plt.title('u2 value: '+ str(u2_value[0][pred_index]) , fontsize=10)
    plt.ylabel(pred_var)
    plt.xlabel('Days')
    plt.grid(False)
    plt.autoscale(axis='x', tight=True)
    plt.legend(loc='upper left', frameon=False)
    fig_path = results_folder + 'preds_day_' + str(pred_index) + "_last_" + str(show_last)
    plt.savefig(fig_path)
    #plt.show()
    plt.close('all')



def plot_sample_prediction(test_x, test_y, test_preds, pred_no, tw, fut_pred, output_size, results_folder, pred_var):
    # original test data

    test_input = test_x[pred_no,:,0]
    test_target = test_y[pred_no, :, 0]
    test_pred = test_preds[pred_no, :, 0]

    plt.title('LSTM Prediction No: ' + str(pred_no))
    plt.ylabel(pred_var)
    plt.xlabel('Days')
    plt.grid(False)
    plt.autoscale(axis='x', tight=True)

    # plot relevant portion of the input data
    plt.plot(test_input, label='input test data')

    # plot actual data for prediction window
    x = np.arange(tw, tw + fut_pred*output_size, 1)
    plt.plot(x, test_target, label='target data', marker= 'o')

    #plot predictions
    plt.plot(x, test_pred, label='prediction',color='teal', marker= 'o', linestyle='--')
    plt.xlim((0,tw + fut_pred*output_size +1 )) #so we can see the prediction in the plot
    plt.legend(loc='upper left', frameon=False)

    #save
    file_name = 'Test_prediction_no' + str(pred_no)
    plt.savefig(results_folder + file_name)
    #plt.show()
    plt.close('all')





