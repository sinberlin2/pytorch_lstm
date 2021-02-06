import torch
import torch.nn as nn
from torch.autograd import Variable


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
                            num_layers = self.num_layers, dropout=self.dropout )  #nonlinearity is by default tanh, should be able to change to relu

        self.linear = nn.Linear(in_features = self.hidden_layer_size, out_features= self.input_size, bias=True)  #or outputsize size?

        #define hidden cell, not necessary
        self.hidden_cell=None

    def init_hidden(self, batch_size):
        #hidden and cell state have shape (num_layers * num_directions, batch_size, hidden_layer_size)
        hidden_cell = (Variable(torch.zeros((self.num_layers, batch_size, self.hidden_layer_size), dtype=torch.double)),
            Variable(torch.zeros((self.num_layers, batch_size, self.hidden_layer_size), dtype=torch.double)))
        return hidden_cell

    def forward(self, input, states=None, print_hidden=False, stateful_batches=False):

        #if no hidden state is provided, we use the hidden state saved by the model
        if states is None:
            self.hidden_cell = self.hidden_cell #this means the hidden state is automatically updated while the model being trained (and passed on).

        #otherwise we use the provided hidden state
        else:
            self.hidden_cell=states

        if print_hidden == True:
            print('in hiidden: ', torch.norm(self.hidden_cell[0]).item())


        #input has shape [batch_size, seq_len, features]

        #With batch_first= True in model definition,  uses second dim as seq_len dimension
        # ! When I used this with batch-first =True, last hidden state and input state were not the same.
        #change seq shape to (batch_size, seq_len, input_size) where input_size means no of features
        # len_input_seq = input_seq.shape[0] #ie batch_size
        # lstm_in= input_seq.view(len_input_seq, -1, 1)


        # without batch first, uses first dim as seq_len dimension
        #create seq of shape [seq_len, batch_size, input_size], so we have to transpose now
        input_seq_b2 = input.transpose(0, 1)
        if input.ndim <2:
            lstm_in_b2 = input_seq_b2.unsqueeze(2)
        else:
            lstm_in_b2 =input_seq_b2

        if stateful_batches==True:
            #hidden state and cell state have shape (num_layers, batchsize,  hidden_layer_size
            #select the last entry of both of the states.
            #We could also just initialise the hidden state for a batch size of 1
            (h,c)=self.hidden_cell
            h = self.hidden_cell[0][:self.num_layers, -1, :self.hidden_layer_size]
            h = h.reshape(h.size(0), 1, h.size(1))
            c = self.hidden_cell[1][:self.num_layers, -1, :self.hidden_layer_size]
            c = c.reshape(c.size(0), 1, c.size(1))

            lstm_out_per_seq=[]
            hidden_states=[]
            cell_states=[]
            # goes through each of the sequences in the batch
            for i, input_tw in enumerate(input):
                #transform input shape to  [seq_len, batch_size=1, input_size]
                lstm_in_it = input_tw.unsqueeze(1)  #adds a dimension for the batch size
                lstm_out, (h,c) = self.lstm(lstm_in_it, (h, c))  # updated hidden state is passed on to the next sequence in the batch.
                lstm_out_per_seq.append(lstm_out)
                hidden_states.append(h)
                cell_states.append(c)

            #record all the last hidden states from all the sequences in batch so
            h_batch=torch.cat(hidden_states,dim=1)
            c_batch=torch.cat(cell_states, dim=1 )
            self.hidden_cell=(h_batch, c_batch)  #the final hidden cell has shape  (num_layers, 1, num_hidden)
            #print(h_batch.shape, 'hiddenstate')

            #here all the hidden states from along the sequences are saved - so train_window * hidden states per sequence
            lstm_out_batch = torch.cat(lstm_out_per_seq, 1)
            linear_in = lstm_out_batch[-1]


        else:
            #print(lstm_in_b2.shape, 'lstminshape')  #[seq_len, batch_size, input_size]
            lstm_out,self.hidden_cell= self.lstm(lstm_in_b2, self.hidden_cell)  # lstmout has shape [seq_len, batch_size, input_size]
            linear_in = lstm_out[-1]  # [batch_size, input_dim (1)]

        if print_hidden == True:
            print('out hiidden: ', torch.norm(self.hidden_cell[0]).item())

        # lstm_output has shape [batch_size, seq_len, hidden_size]

        # Push the output of last step through linear layer; returns (batch_size, 1)
        # linear_in should be [batch_size, input_size]

        #if using batch_first=True, we have to reshape
        ##lstm_out=lstm_out.reshape(self.seq_len, self.batch_size, self.hidden_layer_size) # [seq_len, batch_size, hidden_szie]  # we need the output like this. maybe we can change the input with the batches easier..
        #lstm_out=lstm_out.reshape(lstm_out.size(1), lstm_out.size(0), lstm_out.size(2))

        # #without batch_first- no reshape needed
        # linear_in=lstm_out[-1]   # [batch_size, input_dim (1)]

        # #linear in should be equivalent to last hidden state. can use this to check it.
        # last_hidden= last_hidden_state[-1]
        # print(last_hidden[0], linear_in[0])

        # ###to use all hidden states as input for the linear layer (the sequences must always have the same length)
        # linear_in=lstm_out  # [batch_size, seq_len, input_len]

        #Linear input shape (batch_dim, (seq_lengnth), input_length(no_features))
        linear_out = self.linear(linear_in)

        #linear_out shape is [batch_size, input_size (nofeatures)] #[64,1]
        predictions = linear_out

        return predictions, self.hidden_cell #(last_hidden_state, last_cell_state) for every sequence in batch. unless we pass on the hidden state between sequences where we only pass on the last hidden state. or shall we record them





