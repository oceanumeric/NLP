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
from torch.nn import Parameter
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# parameters are based on this: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html?highlight=lstms


class LSTM(nn.Module):
    """
    LSTM model with only one layer, whereas in PyTorch, the default is two layers.
    """

    def __init__(self, data_path, batch_size, seq_length, hidden_size, drop_prob=0.5):
        super().__init__()

        self.batch_size = batch_size
        self.seq_length = seq_length
        self.hidden_size = hidden_size

        print("::::::::::---------Loading data------::::::::::\n")

        self.text = self.load_data(data_path)
        self.chars = sorted(set(self.text))
        print("Unique characters: ", len(self.chars))
        print(self.text[:100])
        print("::::::::::---------Processing data------::::::::::\n")
        self.encoded_text = self.process_data()

        self.vocab_size = len(self.chars)

        # set up parameters
        # x.shape = (batch_size, seq_length, vocab_size)
        # as we are using one-hot encoding
        # the shape of weight matrices are (vocab_size, hidden_size)
        # as we are following the Pytorch implementation
        self.W_ii = Parameter(torch.Tensor(self.vocab_size, self.hidden_size))
        self.b_ii = Parameter(torch.Tensor(self.hidden_size))
        self.W_hi = Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.b_hi = Parameter(torch.Tensor(self.hidden_size))
        self.W_if = Parameter(torch.Tensor(self.vocab_size, self.hidden_size))
        self.b_if = Parameter(torch.Tensor(self.hidden_size))
        self.W_hf = Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.b_hf = Parameter(torch.Tensor(self.hidden_size))
        self.W_ig = Parameter(torch.Tensor(self.vocab_size, self.hidden_size))
        self.b_ig = Parameter(torch.Tensor(self.hidden_size))
        self.W_hg = Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.b_hg = Parameter(torch.Tensor(self.hidden_size))
        self.W_io = Parameter(torch.Tensor(self.vocab_size, self.hidden_size))
        self.b_io = Parameter(torch.Tensor(self.hidden_size))
        self.W_ho = Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.b_ho = Parameter(torch.Tensor(self.hidden_size))

        # collect all the parameters in a list
        self.params = [
            self.W_ii,
            self.b_ii,
            self.W_hi,
            self.b_hi,
            self.W_if,
            self.b_if,
            self.W_hf,
            self.b_hf,
            self.W_ig,
            self.b_ig,
            self.W_hg,
            self.b_hg,
            self.W_io,
            self.b_io,
            self.W_ho,
            self.b_ho,
        ]

        # initialize the parameters
        print("::::::::::---------Initializing weights------::::::::::\n")
        self.init_weights()

        # add dropout layer
        self.dropout_layer = nn.Dropout(p=drop_prob)

        # add linear layer
        self.lineary_layer = nn.Linear(self.hidden_size, self.vocab_size)

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.params:
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, h, c):
        """
        x.shape = (batch_size, seq_length, vocab_size)
        h.shape = (batch_size, hidden_size); initial hidden state
        c.shape = (batch_size, hidden_size); initial cell state
        """
        batch_size, seq_length, vocab_size = x.shape

        # initialize hidden states and cell states for all the time steps
        outputs = []
        hidden_states = []
        cell_states = []


        # loop through all the time steps
        for t in range(seq_length):
            # extract the input for the current time step
            x_t = x[:, t, :]
            # x_t.shape = (batch_size, vocab_size)
            # h.shape = (batch_size, hidden_size)
            # calculate the input gate
            i_t = torch.sigmoid(x_t @ self.W_ii + self.b_ii + h @ self.W_hi + self.b_hi)
            # calculate the forget gate
            f_t = torch.sigmoid(x_t @ self.W_if + self.b_if + h @ self.W_hf + self.b_hf)
            # calculate the output gate
            # g is cell gate and o is output gate
            g_t = torch.tanh(x_t @ self.W_ig + self.b_ig + h @ self.W_hg + self.b_hg)
            o_t = torch.sigmoid(x_t @ self.W_io + self.b_io + h @ self.W_ho + self.b_ho)
            # calculate the cell state
            # here * means element-wise multiplication
            c = f_t * c + i_t * g_t
            # calculate the hidden state
            h = o_t * torch.tanh(c)
            # append the hidden state and cell state to the list
            # h, c.shape = (batch_size, hidden_size)
            # we need to add a new dimension to the tensor
            outputs.append(o_t.unsqueeze(1))
            hidden_states.append(h.unsqueeze(1))
            cell_states.append(c.unsqueeze(1))

        # now combine all the hidden states and cell states and make it
        # having shape = (batch_size, seq_length, hidden_size)
        outputs = torch.cat(outputs, dim=1)
        hidden_states = torch.cat(hidden_states, dim=1)
        cell_states = torch.cat(cell_states, dim=1)

        # reshape the outputs to (batch_size * seq_length, hidden_size)
        outputs = outputs.reshape(batch_size * seq_length, self.hidden_size)
        hidden_states = hidden_states.reshape(batch_size * seq_length, self.hidden_size)
        cell_states = cell_states.reshape(batch_size * seq_length, self.hidden_size)

        # apply dropout to the outputs
        outputs = self.dropout_layer(outputs)

        # apply linear layer to the outputs
        outputs = self.lineary_layer(outputs)

        # return the outputs, final hidden states and final cell states
        return outputs, h, c

    def init_states(self, batch_size):
        h_0 = torch.zeros(batch_size, self.hidden_size)
        c_0 = torch.zeros(batch_size, self.hidden_size)
        # move to device
        h_0 = h_0.to(device)
        c_0 = c_0.to(device)

        return h_0, c_0

    def train_model(self, epochs, lr=0.001, clip=5):
        # set the model to train mode
        self.train()

        # initialize loss list and optimizer
        losses = []
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        for epoch in tqdm(range(epochs)):
            # initialize hidden and cell states
            h_t, c_t = self.init_states(self.batch_size)

            # loop through all the batches
            for x, y in self.create_batches(self.batch_size, self.seq_length):
                # move the data to device
                x = x.to(device)
                y = y.to(device)

                # create one-hot encoded vectors from the input
                input = F.one_hot(x, self.vocab_size).float()

                # zero out the gradients
                optimizer.zero_grad()

                # forward pass
                output, h_t, c_t = self.forward(input, h_t, c_t)

                # reshape y to (batch_size * seq_length)
                # we need to do this because the loss function expects 1-D input
                targets = y.reshape(self.batch_size * self.seq_length).long()

                # calculate the loss
                loss = F.cross_entropy(output, targets)

                # backward pass
                loss.backward(retain_graph=True)

                # clip the gradients
                nn.utils.clip_grad_norm_(self.parameters(), clip)

                # update the weights
                optimizer.step()

                # append the loss to the losses list
                losses.append(loss.item())

            # print the loss after every 10 epochs
            if (epoch + 1) % 10 == 0:
                print("Epoch: {}, Loss: {:.4f}".format(epoch, losses[-1]))

            # add early stopping
            if losses[-1] < 0.05:
                print("Loss is too low, stopping the training")
                break

        return losses

    def load_data(self, data_path):
        with open(data_path, "r") as f:
            text = f.read()

        return text

    def process_data(self):
        # get all the unique characters
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

        return self.encoded

    def create_batches(self, batch_size, seq_length):
        num_batches = len(self.encoded_text) // (batch_size * seq_length)

        # clip the data to get rid of the remainder
        xdata = self.encoded_text[: num_batches * batch_size * seq_length]
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
            yield xyield, yyield
    
    def predict(self, char, h=None, c=None, top_k=None):


        if h is None or c is None:
            h, c = self.init_states(1)

        # convert the character to one-hot encoded vector
        char_int = torch.tensor([[self.char_to_int[char]]])
        # push to device
        char_int = char_int.to(device)
        # convert to one-hot encoded vector
        char_one_hot = F.one_hot(char_int, self.vocab_size).float()

        # forward pass
        with torch.no_grad():
            output, h, c = self.forward(char_one_hot, h, c)

        # get the character probabilities
        p = F.softmax(output, dim=1).data
        # p.shape = (1, vocab_size)

        if top_k is None:
            # reshape p as (vocab_size)
            p = p.reshape(self.vocab_size)
            # sample with torch.multinomial
            char_next_idx = torch.multinomial(p, num_samples=1)
            # char_next_idx.shape = (1, 1)
            # convert the index to character
            char_next = self.int_to_char.get(char_next_idx.item())
        else:
            # since we have many characters,
            # it is better to use torch.topk to get the top k characters
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

        return char_next, h, c
    
    def sample(self, prime_char="M", h=None, c=None, length=100, top_k=5):

        # initialize the hidden state and cell state
        if h is None or c is None:
            h, c = self.init_states(1)

        # initialize the prime string
        chars = prime_char

        for i in range(length):
            # predict the next character
            char, h, c = self.predict(chars[-1], h, c, top_k=top_k)
            # append the predicted character to the string chars
            chars += char

        return chars

    

if __name__ == "__main__":
    print("Hello world")
    print("Device: ", device)
    print(os.getcwd())

    data_path = "data/sonnets.txt"
    batch_size = 128
    seq_length = 100
    hidden_size = 512
    epoches = 600  # 600 is enough
    learning_rate = 0.001

    lstm = LSTM(data_path, batch_size, seq_length, hidden_size)

    # print out the number of parameters
    print("Number of parameters: {}".format(sum(p.numel() for p in lstm.parameters())))

    # push the model to device
    lstm.to(device)

    # train the model
    losses = lstm.train_model(epoches, learning_rate)

    # generate some text
    print(lstm.sample(prime_char="M", length=500, top_k=5))

# %%
