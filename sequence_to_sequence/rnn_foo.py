# %%
import os
import math
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm


### --------- environment setup --------- ###
# set up the data path
# DATA_PATH = "../data"

# function for setting seed
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
# set up seed globally and deterministically
set_seed(76)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



### --------- define our class module --------- ###
class SimpleRNN(nn.Module):
    """
    We will use a simple RNN to predict the next character in a sequence of characters.
    The input is a sequence of characters, and the output is a sequence of characters. \\
    it will have 2 layers: \\
        1. a simple RNN layer \\
        2. a fully connected layer
    Reference: https://youtu.be/ySEx_Bqxvvo
    """

    def __init__(self, seq_size, batch_size, hidden_size, drop_prob=0.5):
        super(SimpleRNN, self).__init__()

        self.seq_size = seq_size  # later we also use seq_length, it is the same
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.drop_prob = drop_prob

        # load the data
        print("::::::::::---------Loading data------::::::::::\n")
        self.load_data("./data/input.txt")
        print("::::::::::---------Processing data------::::::::::\n")
        self.process_data()
        print("::::::::::---------Creating batches------::::::::::\n")
        self.create_batches(seq_size, batch_size)

        # we will not use embedding layer
        # as we will use one-hot encoding

        # initialize the parameters
        # the first layer is a simple RNN layer
        # h_t = tanh(W_hh * h_{t-1} + W_xh * x_t)
        self.W_xh = nn.Parameter(torch.randn(self.vocab_size, hidden_size, requires_grad=True, device=device))
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size, requires_grad=True, device=device))
        self.b_h = nn.Parameter(torch.zeros(hidden_size, requires_grad=True, device=device))

        # we could initialize the parameters with xavier initialization
        # in this case xavier is better than kaiming
        nn.init.xavier_normal_(self.W_xh)
        nn.init.xavier_normal_(self.W_hh)
        # nn.init.xavier_normal_(self.b_h)

        # the second layer is a fully connected layer
        # y_t = W_hy * h_t
        # we will use Linear layer
        # but we will add a dropout layer before the Linear layer
        self.dropout_layer = nn.Dropout(drop_prob)
        self.linear_layer = nn.Linear(hidden_size, self.vocab_size)

    
    def parameters(self):
        """
        return all the parameters
        """
        params = [self.W_xh, self.W_hh, self.b_h]
        params += list(self.linear_layer.parameters())
        params += list(self.dropout_layer.parameters())
        # print the number of parameters
        print(f"Number of parameters: {sum(p.numel() for p in params)}")
        return params


    def forward(self, x, h):
        """
        x: the input, a sequence of characters with shape (batch_size, seq_size)
        h: the hidden state, a vector with shape (batch_size, hidden_size)
        """
        # initialize the output
        output = []

        # encode the input as one-hot, works like embedding layer
        # it is important to use float() here
        x_one_hot = F.one_hot(x, self.vocab_size).float()
        # x.shape = (batch_size, seq_size, vocab_size)

        # loop through the sequence
        for t in range(self.seq_size):
            # get the current input
            x_t = x_one_hot[:, t, :]
            # x_t.shape = (batch_size, vocab_size)

            # calculate the hidden state
            h = torch.tanh(x_t @ self.W_xh + h @ self.W_hh + self.b_h)
            # h.shape = (batch_size, hidden_size)

            # add dropout
            h = self.dropout_layer(h)

            # calculate the output
            y = self.linear_layer(h)
            # y.shape = (batch_size, vocab_size)

            # append the output
            output.append(y)
        
        # stack the output, dim=1 means we stack along the seq_size
        output = torch.stack(output, dim=1) 
        # output.shape = (batch_size, seq_size, vocab_size)
        
        # return the output and the hidden state
        return output, h
    

    def init_hidden(self, batch_size):
        """
        initialize the hidden state
        """
        return torch.zeros(batch_size, self.hidden_size)


    def train(self, n_epochs, lr=0.001):
        # initialize the optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # initialize the loss list
        loss_list = []

        # loop through the epochs
        for epoch in tqdm(range(n_epochs)):

            # initialize the hidden state
            h = self.init_hidden(self.batch_size)
            # push the hidden state to the device
            h = h.to(device)
            
            # load bach by batch
            for x, y in self.load_batch():
                # x.shape = (batch_size, seq_size)
                # y.shape = (batch_size, seq_size)
                # x will ben encoded as one-hot in the forward pass

                # zero the gradients
                optimizer.zero_grad()

                # forward pass
                output, h = self.forward(x, h)
                # output.shape = (batch_size, seq_size, vocab_size)
                # h.shape = (batch_size, hidden_size)
                # h will be used as the hidden state for the next batch
                # so we need to detach it from the graph after each batch
                h = h.detach()

                # calculate the loss
                # the loss works like this way in pytorch
                # input_data = torch.randn(4, 10)  # 4 samples, 10 classes
                # labels = torch.LongTensor([2, 5, 1, 9])  # target class indices
                # loss = TF.cross_entropy(input_data, labels)
                # we need to reshape the output and y
                # output.shape = (batch_size * seq_size, vocab_size)
                output = output.reshape(-1, self.vocab_size)
                # y.shape = (batch_size * seq_size)
                y = y.reshape(-1)
                # calculate the loss
                loss = F.cross_entropy(output, y)

                # backward pass
                loss.backward()

                # update the parameters
                optimizer.step()

                # append the loss
                loss_list.append(loss.item())
            
            # print the loss
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss_list[-1]:.4f}")

        # return the loss list for plotting
        return loss_list


    def load_data(self, data_path):
        # the data is in input.txt
        with open(data_path, "r") as f:
            self.text = f.read()

        # print the first 100 characters
        print(self.text[:100])


    def process_data(self):
        # we will focus on the character level
        # we need to convert the text into numbers

        # get all the unique characters
        self.chars = list(set(self.text))
        self.vocab_size = len(self.chars)
        print("Vocabulary size: {}".format(self.vocab_size))

        # create a dictionary to map the characters to integers and vice versa
        self.char_to_int = {ch: i for i, ch in enumerate(self.chars)}
        self.int_to_char = {i: ch for i, ch in enumerate(self.chars)}

        # encode the text, torch.long is int64
        self.encoded = torch.tensor(
            [self.char_to_int[ch] for ch in self.text], dtype=torch.long
        )
        # print the first 100 characters
        print("The text has been encoded and the first 100 characters are:")
        print(self.encoded[:100])


    def create_batches(self, seq_length, batch_size):
        self.seq_length = seq_length
        self.batch_size = batch_size

        # calculate the number of batches
        self.n_batches = len(self.encoded) // (seq_length * batch_size)
        # keep only enough characters to make full batches
        self.encoded = self.encoded[: self.n_batches * batch_size * seq_length]
        # create X and Y
        self.X = self.encoded
        self.Y = torch.roll(self.encoded, -1)
        print(f"The shape of X is {self.X.shape}; and the shape of Y is {self.Y.shape}")
        print(f"The first 10 characters in X are {self.X[:10]}")
        print(f"The first 10 characters in Y are {self.Y[:10]}")

        # reshape X and Y
        self.X = self.X.view(batch_size * self.n_batches, seq_length)
        self.Y = self.Y.view(batch_size * self.n_batches, seq_length)
        print(f"The shape of X is {self.X.shape}; and the shape of Y is {self.Y.shape}")


    def load_batch(self):
        # a batch generator to generate X and Y
        for i in range(0, self.X.shape[0], self.batch_size):
            x = self.X[i : i + self.batch_size, :]
            y = self.Y[i : i + self.batch_size, :]
            # x.shape = (batch_size, seq_size)
            # print(f"The shape of x is {x.shape}; and the shape of y is {y.shape}")
            # print(f"The first 10 characters in x are {x[0, :10]}")
            # print(f"The first 10 characters in y are {y[0, :10]}")
            # push x and y to the GPU
            x = x.to(device)
            y = y.to(device)
            yield x, y


    # function to predict the next character
    def predict(self, char, h=None, top_k=None):
        pass # TODO



if __name__ == "__main__":
    print("Hello World!")
    print(os.getcwd())
    print(f"The device is {device}")

    # define the hyperparameters
    seq_length = 100
    batch_size = 128
    hidden_size = 256
    epochs = 50
    learning_rate = 0.001

    # create the model
    model = SimpleRNN(seq_length, batch_size, hidden_size)

    # push the model to the GPU
    model.to(device)

    # train the model
    loss_list = model.train(epochs, learning_rate)

 
# %%
