import torch
import torch.nn as nn
from torch.autograd import Variable


class LSTM(nn.Module):
    def __init__(self, batch_size, input_size, seq_len, hidden_layer_size,  num_layers ,output_size, dropout):
        super(LSTM, self).__init__()
        self.input_size = input_size  #no of features
        self.seq_len=seq_len  # training window
        self.hidden_layer_size = hidden_layer_size
        self.num_layers=num_layers
        self.batch_size=batch_size
        self.output_size=output_size #is 1, model outputs 1 value at a time
        self.batch_first= False
        self.dropout=dropout

        self.lstm = nn.LSTM(input_size = self.input_size,
                            hidden_size = self.hidden_layer_size,
                            num_layers = self.num_layers, batch_first = self.batch_first, dropout=self.dropout )  #nonlinearity is by default tanh, should be able to change to relu

        self.linear = nn.Linear(in_features = self.hidden_layer_size , out_features= self.input_size *self.output_size, bias=True)

        #define hidden cell, not necessary
        self.hidden_cell=None

    def init_hidden(self, batch_size):
        #hidden and cell state have shape (num_layers * num_directions, batch_size, hidden_layer_size)
        hidden_cell = (Variable(torch.zeros((self.num_layers, batch_size, self.hidden_layer_size), dtype=torch.double)),
            Variable(torch.zeros((self.num_layers, batch_size, self.hidden_layer_size), dtype=torch.double)))

        return hidden_cell


    def forward(self, input, states=None, print_hidden=False, stateful_batches=False):

        if states is None:
            self.hidden_cell= self.init_hidden(input.shape[0]) #not batch size, cause in test mode we want to test 1 sequence at a time
        else:
            self.hidden_cell=states

        #input shape is [B, TW, F}
        #create seq of shape [train_window, batch_size, features], so we have to transpose now
        #(we could also use batch_first=True and not transpose)

        input_transposed = input.transpose(0, 1)
        if input.ndim < 2:
            lstm_in = input_transposed.unsqueeze(2)
        else:
            lstm_in = input_transposed

        if stateful_batches is not True:
            lstm_out, self.hidden_cell = self.lstm(lstm_in, self.hidden_cell)
            # lstm_output has shape [seq_len, B, hidden_size]

        else:  #if stateful batches is true,

            # select the last entry of both of the states
            # hidden state and cell state have shape (num_layers, batchsize,  hidden_layer_size)
            # We could also just initialise the hidden state for a batch size of 1. But then we couldnt do batch normalisation on the hidden state if we wanted to

            h = self.hidden_cell[0][:self.num_layers, -1, :self.hidden_layer_size]
            h = h.reshape(h.size(0), 1, h.size(1))
            c = self.hidden_cell[1][:self.num_layers, -1, :self.hidden_layer_size]
            c = c.reshape(c.size(0), 1, c.size(1))

            lstm_out_per_seq = []
            hidden_states = []
            cell_states = []
            # goes through each of the sequences in the batch
            for i, input_tw in enumerate(input):
                # transform input shape to  [seq_len, batch_size=1, input_size]
                lstm_in_it = input_tw.unsqueeze(1)  # adds a dimension for the batch size
                lstm_out, (h, c) = self.lstm(lstm_in_it, (h, c))  # updated hidden state is passed on to the next sequence in the batch.
                lstm_out_per_seq.append(lstm_out)
                hidden_states.append(h)
                cell_states.append(c)

            # record all the last hidden states from all the sequences in batch so
            h_batch = torch.cat(hidden_states, dim=1)
            c_batch = torch.cat(cell_states, dim=1)
            self.hidden_cell = (h_batch, c_batch)  # the final hidden cell has shape  (num_layers, 1, num_hidden)

            # here all the hidden states from along the sequences are saved - so train_window * hidden states per sequence
            lstm_out_batch = torch.cat(lstm_out_per_seq, dim=1)
            linear_in = lstm_out_batch[-1]


        # Push the output of last step through linear layer (default behaviour of keras); linear_in should be [batch_size, hidden_layer_size]; returns (batch_size, F)
        linear_in=lstm_out[-1]


        #predictions; linear_out shape is [batch_size, F]
        linear_out = self.linear(linear_in)

        return linear_out, self.hidden_cell


'''
 It is also possible to use all hidden states as input for the linear layer (the sequences must always have the same length
      then the linear layer probably needs to be initialised as:
      nn.Linear(in_features=self.hidden_layer_size * self.seq_len, out_features=self.input_size * self.output_size, bias=True)
      lstm_out=lstm_out.reshape(lstm_out.shape[1], lstm_out.shape[0], lstm_out.shape[2])
      linear_in =lstm_out.contiguous().view(lstm_out.shape[0], -1)

      #linear in should be equivalent to last hidden state. last_hidden= last_hidden_state[-1]
      print(self.hidden_cell[0][-1], linear_in)  #to check this

      '''

