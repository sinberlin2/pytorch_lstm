# Pytorch LSTM with different options of initialisation

### Initialisation of states in the LSTM:
The hidden state and the cell state are initialised with zeros at the beginning of training, rather than a random initialisation, as the states’ values modify which values in the parts of the new inputs contribute to the new states. A zero-initialization therefore means that the weights do not contain any information, which makes sense as nothing has been learnt before training. The states should  be initialised at the beginning of each training epoch, to prevent the model to rely on the previous training epochs for a certain performance and therefore ensuring generalisability.

### Methods for Initialisation of states 
The most common method of initialising states in the LSTM is to initialise them at every batch. This is the default behaviour in keras.
This means that there is no learning between the batches other than through the updating of weights. This strategy improves the generalisability of the model.
Here I have also implemented two other options. The first one is to initialise the states at the beginning of each epoch and states are passed on between batches. This method is called stateful \cite{yadav2020optimizing} and can be used with keras by specifying stateful= True. 
This would in theory allow learning along the whole sequence rather than only along each of the sequences in the batch. However, depending on how big the batches are, the gap between the hidden states that would be transferred would not be that beneficial for model performance. 
To overcome this problem, another form of state initialisation was implemented where the hidden states are passed on between the sequences within each batch. While we saw some performance improvement with this method, it comes at the expense of the increased speed when using batches where batch sequences are processed in parallel. Passing on the sequence between batches or sequences in batches decreases model generalisability.

### TLDR
1. Initialisation at every batch:
Every sequence is a clean sheet. Every sequence tested starts with a zero hidden state
However there is no learning between batches. This method is only useful if sequence length and batch size are large enough to learn enough about sequence distribution by themselves. 

2. Intitialisation at every epoch, Stateful between Batches:
Hidden state are passed on between batches. The last state for each sample at index i in a batch j will be used as initial state for the sample of index i in the following batch j+1. 
 - advantages: faster due to parallel processing of sequences.
 - disadvantages: Sequence Xi receives hidden state from Sequence Xi-batch_size. Optimally, it should receive hidden state Xi-1, but there should be some relevant information in the hidden state from Sequence Xi-batch_size (better than nothing).

3. Stateful within and between batches:
Within 1 batch, hiddens state is passed on from sequence to sequence.
Then backpropagation on whole batch (and all hidden states)
Input for the next batch is only the last hidden state
In theory: Allows to optimally learn dependencies between sequences
However in practice, we dont see the advantage
If there is no benefit, then there really is too much noise/ no signal
   -  hidden state is passed on from sequence to sequence within batch and to the first sequence in the following batch.


### Parameters for the different Options
1. Initialise at every batch. - stateful = False 
2. Initialise at every epoch. Stateful between batches. -  stateful = True , stateful_batches = False
3. Initialise at every epoch. Stateful within and between batches. stateful = True , stateful_batches = True



#https://stackabuse.com/time-series-prediction-using-lstm-with-pytorch-in-python/
