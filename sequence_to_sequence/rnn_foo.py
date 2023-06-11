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
# def set_seed(seed):
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)


# # set up seed globally and deterministically
# set_seed(76)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

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

    def __init__(self, data_path, seq_size, batch_size, hidden_size, drop_prob=0.5):
        super(SimpleRNN, self).__init__()

        self.seq_size = seq_size  # later we also use seq_length, it is the same
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.drop_prob = drop_prob

        # load the data
        print("::::::::::---------Loading data------::::::::::\n")
        self.load_data(data_path)
        print("::::::::::---------Processing data------::::::::::\n")
        self.process_data()
        print(f"The size of the vocabulary is {self.vocab_size}")
        print("::::::::::---------Creating batches------::::::::::\n")
        self.create_batches(seq_size, batch_size)

        # we will not use embedding layer
        # as we will use one-hot encoding

        # initialize the parameters
        # the first layer is a simple RNN layer
        # h_t = tanh(W_hh * h_{t-1} + W_xh * x_t)
        self.W_xh = nn.Parameter(
            torch.randn(self.vocab_size, hidden_size, device=device)
        )
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size, device=device))
        self.b_h = nn.Parameter(torch.zeros(hidden_size, device=device))

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

    def forward(self, x, h):
        """
        x: the input, a sequence of characters with shape (batch_size, seq_size)
        h: the hidden state, a vector with shape (batch_size, hidden_size)
        """
        # initialize the hidden output
        hidden_output = []

        # one could also do output = []
        # and then calculate y inside the loop
        # and append y to output

        # encode the input as one-hot, works like embedding layer
        # it is important to use float() here
        x_one_hot = F.one_hot(x, self.vocab_size).float()
        # x.shape = (batch_size, seq_size, vocab_size)

        # loop through the sequence
        for t in range(x_one_hot.shape[1]):
            # get the current input
            x_t = x_one_hot[:, t, :]
            # x_t.shape = (batch_size, vocab_size)

            # calculate the hidden state
            h = torch.tanh(x_t @ self.W_xh + h @ self.W_hh + self.b_h)
            # h.shape = (batch_size, hidden_size)

            # add dropout, drop out should be applied to the hidden state only
            # h = self.dropout_layer(h)

            # calculate the output
            # this method is slower
            # y = self.linear_layer(h)

            # append the output
            hidden_output.append(h)

        # stack the output, dim=1 means we stack along the seq_size
        # output = torch.stack(output, dim=1)
        # output.shape = (batch_size, seq_size, vocab_size)
        # now stack the hidden_output along the seq_size
        hidden_output = torch.stack(hidden_output, dim=1)
        # hidden_output.shape = (batch_size, seq_size, hidden_size)

        # drop out should be applied to the hidden state only
        hidden_output = self.dropout_layer(hidden_output)

        # calculate the output
        output = self.linear_layer(hidden_output)

        # return the output and the hidden state
        # we are returnning h because we will use it in the next iteration
        # as each cell in the sequence depends on the previous cell
        # not the whole batch and the whole sequence
        # we are not returning the hidden_output because we will not use it
        return output, h

    def init_hidden(self, batch_size):
        """
        initialize the hidden state with next because we will use it
        in the next iteration
        """
        return torch.zeros(batch_size, self.hidden_size)

    def train_model(self, n_epochs, lr=0.001):
        # initialize the optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # initialize the loss function
        loss_fn = nn.CrossEntropyLoss()

        # initialize the loss list
        self.loss_list = []

        # loop through the epochs
        for epoch in tqdm(range(n_epochs)):
            # initialize the hidden state
            h = self.init_hidden(self.batch_size)
            # push the hidden state to the device
            h = h.to(device)

            # load bach by batch
            for x, y in self.create_batches(self.batch_size, self.seq_size):
                # x.shape = (batch_size, seq_size)
                # y.shape = (batch_size, seq_size)
                # x will ben encoded as one-hot in the forward pass

                # zero the gradients
                self.zero_grad()

                # forward pass
                output, h = self.forward(x, h)
                # output.shape = (batch_size, seq_size, vocab_size)
                # h.shape = (batch_size, hidden_size)
                # h will be used as the hidden state for the next batch
                # so we need to detach it from the graph after each batch

                # calculate the loss
                # the loss works like this way in pytorch
                # input_data = torch.randn(4, 10)  # 4 samples, 10 classes
                # labels = torch.LongTensor([2, 5, 1, 9])  # target class indices
                # loss = TF.cross_entropy(input_data, labels)
                # we need to reshape the output and y
                # output.shape = (batch_size * seq_size, vocab_size)
                output = output.reshape(-1, self.vocab_size)
                # y.shape = (batch_size * seq_size)
                y = y.reshape(self.batch_size * self.seq_size).long().to(device)
                # calculate the loss
                loss = loss_fn(output, y)

                # retain the graph if you want to use it later
                # loss.retain_grad()

                # backward pass
                loss.backward(retain_graph=True)

                # clip the gradients
                # this is very important !!!
                # otherwise the gradients will explode
                # this is a very common technique in RNN
                # but it is not used in the original paper
                nn.utils.clip_grad_norm_(self.parameters(), 5)

                # update the parameters
                optimizer.step()

                # append the loss
                self.loss_list.append(loss.item())

                # early stopping
                # if loss.item() < 1.21:
                #     print("Loss is less than 1.21, stop training")
                #     return loss_list

            # print the loss
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {self.loss_list[-1]:.4f}")

        # return the loss list for plotting
        return self.loss_list

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

        # print the dictionaries
        print(self.char_to_int)

        # encode the text, torch.long is int64
        self.encoded = torch.tensor(
            [self.char_to_int[ch] for ch in self.text], dtype=torch.long
        )
        # print out the length
        print("Text length: {}".format(self.encoded.size(0)))
        # print the first 100 characters
        print("The text has been encoded and the first 100 characters are:")
        print(self.encoded[:100])

    def create_batches(self, batch_size, seq_length):
        num_batches = len(self.encoded) // (batch_size * seq_length)

        # clip the data to get rid of the remainder
        xdata = self.encoded[: num_batches * batch_size * seq_length]
        ydata = torch.roll(xdata, -1)

        # reshape the data
        # this step is very important, because we need to make sure
        # the input and targets are aligned
        # we need to make sure xdata.shape = (batch_size, seq_length*num_batches)
        # because we feed one batch at a time
        xdata = xdata.view(batch_size, -1)
        ydata = ydata.view(batch_size, -1)

        # now we will divide the data into batches
        for i in range(0, xdata.size(1), seq_length):
            xyield = xdata[:, i : i + seq_length]
            yyield = ydata[:, i : i + seq_length]
            # push the data to the GPU
            xyield = xyield.to(device)
            yyield = yyield.to(device)
            yield xyield, yyield

    # function to predict the next character based on character
    def predict(self, char, h=None, top_k=None):
        # assume char is a single character
        # convert the character to integer
        char = torch.tensor([[self.char_to_int[char]]])
        # push the character to the GPU
        char = char.to(device)
        # reshape char as (1, 1)
        char = char.reshape(1, 1)

        # initialize the hidden state
        if h is None:
            # h.shape = (1, hidden_size)
            # because we only have one character
            h = torch.zeros((1, self.hidden_size))

        # push the hidden state to the GPU
        h = h.to(device)

        # call the model to get the output and hidden state
        with torch.no_grad():
            # get the output and hidden state
            output, h = self(char, h)
            # output.shape = (1, vocab_size)

        # get the probabilities
        # dim=1 because we want to get the probabilities for each character
        p = F.softmax(output, dim=1).data

        # if top_k is None, we will use torch.multinomial to sample
        # otherwise, we will use torch.topk to get the top k characters
        if top_k is None:
            # reshape p as (vocab_size)
            p = p.reshape(self.vocab_size)
            # sample with torch.multinomial
            char_next_idx = torch.multinomial(p, num_samples=1)
            # char_next_idx.shape = (1, 1)
            # convert the index to character
            char_next = self.int_to_char.get(char_next_idx.item())
        else:
            p, char_next_idx = p.topk(top_k)
            # char_next_idx.shape = (1, top_k)
            # convert the index to character
            char_next_idx = char_next_idx.squeeze().cpu().numpy()
            # char_next_idx.shape = (top_k)
            # randomly select one character from the top k characters
            p = p.squeeze().cpu().numpy()
            # p.shape = (top_k)
            char_next_idx = np.random.choice(char_next_idx, p=p / p.sum())
            # char_next_idx.shape = (1)
            # convert the index to character
            char_next = self.int_to_char.get(char_next_idx.item())

        return char_next, h

    # function to generate text
    def generate_text(self, char="a", h=None, length=100, top_k=None):
        # intialize the hidden state
        if h is None:
            h = torch.zeros((1, self.hidden_size))
        # push the hidden state to the GPU
        h = h.to(device)

        # initialize the generated text
        gen_text = char

        # predict the next character until we get the desired length
        # we are not feedding the whole sequence to the model
        # but we are feeding the output of the previous character to the model
        # because the the memory was saved in the hidden state
        for i in range(length):
            char, h = self.predict(char, h, top_k)
            gen_text += char

        return gen_text


# debug process:
# 1. check the shape of all inputs and outputs
# 2. change the weight initialization to xavier_uniform
# 3. apply dropout to y instead of h
# 4. add gradient clipping
# try different epochs and learning rate
# fix the bug in the predict function
# with epochs = 100 and learning_rate = 0.001, the loss is 2.09
# for epochs = 300 and learning_rate = 0.001, the loss is 1.93 and it takes 12 minutes

if __name__ == "__main__":
    print("Hello World!")
    print(os.getcwd())
    print(f"The device is {device}")

    # define the hyperparameters
    seq_length = 100
    batch_size = 128  # 512
    hidden_size = 512  # or 256
    epochs = 100
    learning_rate = 0.001

    # notes: batch_size means how man samples we feed to the model at one time
    # hidden_size means the number of hidden units in the RNN cell
    # if batch_size * hidden_size is very large, then each forward pass will take a long time

    text_file = "data/sonnets.txt"

    # create the model
    model = SimpleRNN(text_file, seq_length, batch_size, hidden_size)

    # print out number of parameters
    print(f"Number of parameters is {sum(p.numel() for p in model.parameters())}")
    # input.txt = 329 281
    # sonnets.txt = 325181

    # # push the model to the GPU
    model.to(device)

    # # train the model
    model.train()
    loss_list = model.train_model(epochs, learning_rate)

    # generate text
    print(model.generate_text(char="H", length=100, top_k=5))


# %%
