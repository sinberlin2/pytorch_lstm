import pandas as pd
import numpy as np # linear algebra
import os.path
from os import path
import torch
import utils #for loss function and u2 value

##Evaluation of the model/ Testing

def test_model(model, test_inout_seq, fut_pred, tw, stateful, init_batch):

    #initialise arrays with shape [B, time steps]
    len_preds=len(test_inout_seq)-fut_pred+1
    test_predictions_all=np.empty(shape=(len_preds,fut_pred))
    naive_predictions_all= np.empty(shape=(len_preds ,fut_pred))
    labels1=  np.empty(shape=(len_preds,fut_pred))
    for pred_day in range(fut_pred):
        labels1[:,pred_day] = [item[1][:,:,0] for item in test_inout_seq][pred_day: len_preds+pred_day]

    #print(labels1)

    # notify all our layers that we are in eval mode, that way, batchnorm or dropout layers will work in eval mode instead of training mode
    model.eval()

    #dataloader produces pytorch tensors # batch size =1 because testing
    test_loader = torch.utils.data.DataLoader(test_inout_seq, batch_size=1, shuffle=False, num_workers=1, drop_last=False)

    for batch_no, (seq, labels) in enumerate(test_loader):
        seq = seq.reshape(seq.shape[0], seq.shape[1], seq.shape[-1])
        labels = labels.reshape(labels.shape[0], labels.shape[1], labels.shape[-1])

        #initialise list for predictions for each sequence for multiple predictions ahead
        test_losses_seq=0   # naive score, last value of test input is prediction
        naive_losses_seq=0
        naive_truth= seq[:,-1]

        for i in range(fut_pred):
            if batch_no +i < len_preds:  #to make sure that we can get a label for the next prediciton step

                #use only the last values that fit in the sliding window for a prediciton of multiple steps
                seq= seq[:,-tw:,:]  #seq has shape [batch_size=1, sliding_window_size]

                with torch.no_grad():  # reduces time cause no gradients are being calculated
                    if i==0:
                        if init_batch==1:
                            # we reinitialise hidden state, we dont keep memory from before
                            # hidden state and cell state shape are [num_layers, batch_size, hidden_layer_size)
                            model.hidden_cell = (torch.zeros((model.num_layers, 1, model.hidden_layer_size), dtype=torch.float64),
                                                 torch.zeros((model.num_layers,1, model.hidden_layer_size), dtype=torch.float64))
                        else:
                            # We want to pass on the hidden state from the training/validation.
                            # We need to get the states from the last sequence in last batch, as we test one sequence at a time in the evaluation mode.
                            h= model.hidden_cell[0][:model.num_layers, -1, :model.hidden_layer_size]
                            h= h.reshape(h.size(0),  1,h.size(1))
                            c = model.hidden_cell[1][:model.num_layers, -1, :model.hidden_layer_size]
                            c = h.reshape(c.size(0), 1, c.size(1))

                            model.hidden_cell=(h, c)

                        test_prediction, (h, c) = model.forward(seq, model.hidden_cell , print_hidden= False) #, states=None if i put in states=None, it doesnt think it is none anymore   # hidden state and cell state are reset

                    else:
                        print(seq.shape)
                        test_prediction, _ = model.forward(seq) #it's not necessart to pass hidden and cell state since the model is by definition always reusing the hidden state


                    #Add prediction to input to make another step in the prediction
                    seq= torch.cat((seq.squeeze(0), test_prediction)).unsqueeze(0)

                    #record test prediction for each sequence (batch number is sequence and i is the prediction day)
                    test_predictions_all[batch_no, i] = test_prediction[:,0].item()
                    naive_predictions_all[batch_no, i] = naive_truth[:,0]

                    #add the next label value to labels with multiple predictions (change maybe to use labels1)
                    if i > 0 :
                        add_to_label = torch.DoubleTensor(test_inout_seq[batch_no+ i][1])
                        labels = torch.cat((labels.squeeze(0), add_to_label.squeeze(0))).unsqueeze(0)

                    #calculate loss
                    test_loss= utils.loss_function(test_prediction, labels[:,i]).item()
                    test_losses_seq+=test_loss  #or save as list to then extract first and last values

                    #calculate naive loss
                    naive_loss= utils.loss_function(naive_truth, labels[:,i]).item()
                    naive_losses_seq+=naive_loss


    #average loss for each prediction time steps
    test_loss_all = np.mean(test_predictions_all, axis=0)
    naive_loss_all = np.mean(naive_predictions_all, axis=0)

    #avergage loss for all prediciton time steps
    test_loss_avg = np.mean(test_loss_all, axis=0)
    naive_loss_avg= np.mean(naive_loss_all, axis=0)

    print("test_loss_avg: %f" % test_loss_avg)
    print("naive_loss_avg: %f" % naive_loss_avg)

    #calcuate U2 values for each prediction.
    # U2 value should get better with every prediciton time step as the naive prediciton becomes increasingly inaccurate

    print(test_predictions_all[:5])
    print(labels1[:5])
    u2_values=[]
    for pred_day in range(fut_pred):
        u2_values.append(utils.get_u2_value(test_predictions_all[:,pred_day], labels1[:,pred_day], naive_predictions_all[:,pred_day]))

    print(u2_values)
    print("u2_value_first: %f" % u2_values[0])


    return(test_predictions_all,  naive_predictions_all, u2_values)
