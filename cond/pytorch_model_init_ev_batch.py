import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, batch_size, input_size, seq_len, hidden_layer_size,  num_layers ,output_size, dropout):
        super(LSTM, self).__init__()
        self.input_size = input_size  #relates to no of features- has to be one for unconditional model
        self.seq_len=seq_len  # training window
        self.hidden_layer_size = hidden_layer_size
        self.num_layers=num_layers
        self.batch_size=batch_size
        self.output_size=output_size
        self.batch_first= False
        self.dropout=dropout

        self.lstm = nn.LSTM(input_size = self.input_size,
                            hidden_size = self.hidden_layer_size,
                            num_layers = self.num_layers, batch_first = self.batch_first, dropout=self.dropout )  #nonlinearity is by default tanh, should be able to change to relu

        self.linear = nn.Linear(in_features = self.hidden_layer_size, out_features= self.input_size, bias=True)  #or input size?

        #define hidden cell, not necessary
        self.hidden_cell=None


    def forward(self, input, states=None, print_hidden=False):

        if states is None:
            self.hidden_cell = (torch.zeros((self.num_layers, input.size(0), self.hidden_layer_size), dtype=torch.float64),   #(num_layers * num_directions, batch, hidden_size):
                         torch.zeros((self.num_layers, input.size(0),  self.hidden_layer_size), dtype=torch.float64))   # (num_layers * num_directions, batch, hidden_size): #1, 1, model.hidden_layer_size

        else:
            self.hidden_cell=states

        if print_hidden == True:
            print('in hiidden: ', torch.norm(self.hidden_cell[0]).item())


        #input has shape [batch_size, seq_len]

        #With batch_first= True in model definition,  uses second dim as seq_len dimension
        # ! When I used this with batch-first =True, last hidden state and input state were not the same.
        #change seq shape to (batch_size, seq_len, input_size) where input_size means no of features
        # len_input_seq = input_seq.shape[0] #ie batch_size
        # lstm_in= input_seq.view(len_input_seq, -1, 1)


        # without batch first, uses first dim as seq_len dimension
        #create seq of shape [seq_len, batch_size, input_size], so we have to transpose now
        input_seq_b2 = input.transpose(0, 1)
        if input.ndim < 2:
            lstm_in_b2 = input_seq_b2.unsqueeze(2)
        else:
            lstm_in_b2 = input_seq_b2


        #lstm_out,(last_hidden_state, last_cell_state)= self.lstm(lstm_in_b2, states)  #first I had it 1, -1. changes the vorzeichen in output.
        lstm_out,self.hidden_cell= self.lstm(lstm_in_b2, self.hidden_cell)  #first I had it 1, -1. changes the vorzeichen in output.
        # lstm_output has shape [batch_size, seq_len, hidden_size]

        if print_hidden == True:
            print('out hiidden: ', torch.norm(self.hidden_cell[0]).item())



        # Push the output of last step through linear layer; returns (batch_size, 1)
        # linear_in should be [batch_size, input_size]

        #if using batch_first=True, we have to reshape
        ##lstm_out=lstm_out.reshape(self.seq_len, self.batch_size, self.hidden_layer_size) # [seq_len, batch_size, hidden_szie]  # we need the output like this. maybe we can change the input with the batches easier..
        #lstm_out=lstm_out.reshape(lstm_out.size(1), lstm_out.size(0), lstm_out.size(2))

        #without batch_first- no reshape needed
        linear_in=lstm_out[-1]   # [batch_size, input_dim (1)]

        # #linear in should be equivalent to last hidden state. can use this to check it.
        # last_hidden= last_hidden_state[-1]
        # print(last_hidden[0], linear_in[0])

        # ###to use all hidden states as input for the linear layer (the sequences must always have the same length)
        # linear_in=lstm_out  # [batch_size, seq_len, input_len]

        #Linear input shape (batch_dim, (seq_lengnth), input_length(no_features))
        linear_out = self.linear(linear_in)

        #linear_out shape is [batch_size, input_size (nofeatures)] #[64,1]
        predictions = linear_out

        return predictions, self.hidden_cell #(last_hidden_state, last_cell_state)





