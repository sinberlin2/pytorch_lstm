import os.path
import torch
import utils
import pytorch_model_init_ev_batch
import pytorch_model_cond


#TRAINING
# Initialise model and optimiser
def train(train_inout_seq, val_inout_seq, train_window, epochs, batch_size, input_size, hidden_layer_size, num_layers, output_size, lr, lr_decay, dropout, results_folder, stateful, init_batch):
    if init_batch==True:
        print('batches initialised with zeros.')
        model = pytorch_model_init_ev_batch.LSTM(batch_size=batch_size, input_size=input_size, seq_len=train_window,
                                   hidden_layer_size=hidden_layer_size,
                                   num_layers=num_layers, output_size=output_size,
                                   dropout=dropout)
    else:
        model = pytorch_model_cond.LSTM(batch_size=batch_size, input_size=input_size, seq_len=train_window, hidden_layer_size=hidden_layer_size,
                  num_layers=num_layers, output_size=output_size, dropout=dropout)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=lr_decay)
    print(model)
    model=model.double()
    if stateful==1:
        print('Epochs are initialised with zeros, batches are stateful. The hidden state is passed on between the sequences of the batch.')
    else:
        print('Epochs are initialised with zeros, batches are stateless. Hidden states are only passed between batches.')

    #for (mini) batches
    train_loader = torch.utils.data.DataLoader(train_inout_seq, batch_size= batch_size, shuffle=False, num_workers=1, drop_last=True )  #false bc we dont wanna shuffle time
    val_loader = torch.utils.data.DataLoader(val_inout_seq, batch_size= batch_size, shuffle=False, num_workers=1, drop_last=True)

     #initialise variables
    train_losses=[]
    val_losses=[]

    for i in range(epochs):
        epoch_train_loss=0
        epoch_val_loss=0

        # # we initialise hidden state every epoch, to not make the results of the model depend on the previous training results
        model.hidden_cell = model.init_hidden(model.batch_size)

        model.train()

        for step, (seq, labels) in enumerate(train_loader):
            seq=seq.reshape(seq.shape[0], seq.shape[1], seq.shape[-1]).double()  #seq has shape [B, TW, Features]
            labels=labels.reshape(labels.shape[0], labels.shape[-1]).double() #labels has shape  [B, fut-pred steps, features]

            # zero the gradients  #Pytorch accumulates gradients, we need to clear gradients before each instance
            optimizer.zero_grad()

            # forward pass through the model
            if init_batch==1:
                # if the model initialises the hidden state at each batch, we should not input the previous hidden state
                y_pred, model.hidden_cell = model.forward(seq)

            else:
                # Starting each batch, we detach the hidden state from how it was previously produced.
                # This means that we consider the input hidden cell from the last batch as constant
                # and we dont propagate back till the beginning of the input data.
                y_pred, model.hidden_cell = model.forward(seq, (model.hidden_cell[0].detach(), model.hidden_cell[1].detach()), print_hidden=False, stateful_batches=stateful)

            #compute loss
            single_loss =  utils.loss_function(y_pred, labels)

            #check gradients
            check_gradients = False
            if check_gradients == True:
                grads = [torch.norm(param).item() for param in model.parameters()]
                print(grads[0])

            # get gradients with respect to that loss, backward propagation,
            single_loss.backward()

            # # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs., was not useful for our experiments
            #torch.nn.utils.clip_grad_norm(model.parameters(), 1.0) # good clip norm value is 1.0

            # actual optimizing step
            optimizer.step()

            loss_current = float(single_loss.item())
            epoch_train_loss+=loss_current

        for val_batch_no, (seq, labels) in enumerate(val_loader):
            seq=seq.reshape(seq.shape[0], seq.shape[1], seq.shape[-1]).double()  #seq has shape [B, TW, Features]
            labels = labels.reshape(labels.shape[0], labels.shape[-1]).double()  #labels has shape  [B, fut-pred steps, features]

            # we dont have to input the detached hiddens state, cause there is no backprop
            with torch.no_grad():
                # forward pass through the model
                # underscore is for hidden and cell state which we ignore during training.
                if init_batch==1:
                    y_pred, _ = model.forward(seq)
                else:
                    y_pred, _ = model.forward(seq, stateful_batches=stateful)

                #labels = labels.squeeze(1) #or change model output to correct shape
                single_loss = utils.loss_function(y_pred, labels).item()
                epoch_val_loss += single_loss

        epoch_train_loss = epoch_train_loss / len(train_inout_seq)
        train_losses.append(epoch_train_loss)
        epoch_val_loss= epoch_val_loss / len(val_inout_seq)
        val_losses.append(epoch_val_loss)

        if i%1 == 0:
            print(f'epoch: {i:3} val loss: {epoch_val_loss:10.8f}')

    print('Done training.')

    total_train_loss = float((sum(train_losses))/epochs)
    print("total_training_loss: %f" % total_train_loss)
    total_val_loss = float((sum(val_losses))/epochs)
    print("total_val_loss: %f" % total_val_loss)

    # #Plot Training and val loss
    utils.plot_loss(epochs, results_folder, train_losses, val_losses)

    #save model in results folder (can load it from there more easily
    model_path= results_folder + 'LSTM_' + 'tw_' + str(train_window) + '.pt'
    torch.save(model.state_dict(), model_path)
    print('Model saved in ' + model_path)

    return model