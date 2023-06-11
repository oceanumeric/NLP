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


### --------- define our class module --------- ###
class SimpleRNN(nn.Module):
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

        # initialize the hidden state
        self.Wxh = Parameter(torch.Tensor(self.vocab_size, self.hidden_size))
        self.Whh = Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.bh = Parameter(torch.zeros(self.hidden_size))

        nn.init.xavier_uniform_(self.Wxh)
        nn.init.xavier_uniform_(self.Whh)

        # add dropout layer
        self.droupout_layer = nn.Dropout(drop_prob)

        # add the linear layer
        self.lineary_layer = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, x, h_t):
        """
        x: input, shape = (batch_size, seq_length, vocab_size), here we use one-hot encoding
        h_prev: hidden state from previous cell, shape = (batch_size, hidden_size)
        """
        batch_size, seq_length, vocab_size = x.shape
        hidden_states = []

        for t in range(seq_length):
            x_t = x[:, t, :]
            h_t = torch.tanh(x_t @ self.Wxh + h_t @ self.Whh + self.bh)
            # h_t.shape = (batch_size, self.hidden_size)
            # h_t.shape = (batch_size, 1, self.hidden_size)
            # do not do h_t = h_t.unsqueeze(1)
            hidden_states.append(h_t.unsqueeze(1))

        # concatenate the hidden states
        hidden_states = torch.cat(hidden_states, dim=1)

        # reshape the hidden states
        hidden_states = hidden_states.reshape(batch_size * seq_length, self.hidden_size)

        # apply dropout
        hidden_states = self.droupout_layer(hidden_states)

        # apply the linear layer
        logits = self.lineary_layer(hidden_states)

        # logits.shape = (batch_size * seq_length, vocab_size)
        # h_t was unsqueezed, so we need to squeeze it back

        return logits, h_t

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)

    def train_model(self, epochs, lr=0.001, clip=5):
        # set the model to train mode
        self.train()

        loss_list = []

        # define the optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        for epoch in tqdm(range(epochs)):
            # initialize the hidden state
            h_t = self.init_hidden(self.batch_size)
            # push the hidden state to GPU, if available
            h_t = h_t.to(device)

            for x, y in self.create_batches(self.batch_size, self.seq_length):
                # move the data to GPU, if available
                x = x.to(device)
                y = y.to(device)

                # create one-hot encoding
                # do not do x = F.one_hot(x, self.vocab_size).float()
                # it will have a error message: "RuntimeError: one_hot is not implemented for type torch.cuda.LongTensor"
                inputs = F.one_hot(x, self.vocab_size).float()

                # zero out the gradients
                self.zero_grad()

                # get the logits
                logits, h_t = self.forward(inputs, h_t)

                # reshape y to (batch_size * seq_length)
                # we need to do this because the loss function expects 1-D input
                targets = y.reshape(self.batch_size * self.seq_length).long()

                # calculate the loss
                loss = F.cross_entropy(logits, targets)

                # backpropagate
                loss.backward(retain_graph=True)

                # clip the gradients
                nn.utils.clip_grad_norm_(self.parameters(), clip)

                # update the parameters
                optimizer.step()

                # append the loss
                loss_list.append(loss.item())

            # print the loss every 10 epochs
            if epoch % 10 == 0:
                print("Epoch: {}, Loss: {:.4f}".format(epoch, loss_list[-1]))

        return loss_list

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

    # function to predict the next character based on character
    def predict(self, char, h=None, top_k=None):
        # assume char is a single character
        # convert the character to integer
        char = torch.tensor([[self.char_to_int[char]]])
        # push the character to the GPU
        char = char.to(device)
        # one-hot encode the character
        inputs = F.one_hot(char, self.vocab_size).float()

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
            output, h = self(inputs, h)
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


if __name__ == "__main__":
    print("Hello World")
    print(os.getcwd())

    seq_length = 100
    batch_size = 128  # 512
    hidden_size = 512  # or 256
    epochs = 300
    learning_rate = 0.001

    rnn_model = SimpleRNN(
        data_path="data/sonnets.txt",
        batch_size=batch_size,
        seq_length=seq_length,
        hidden_size=hidden_size,
    )
    # print out number of parameters
    print(f"Number of parameters is {sum(p.numel() for p in rnn_model.parameters())}")
    # push to GPU
    rnn_model.to(device)
    # train the model
    loss_list = rnn_model.train_model(epochs=epochs, lr=learning_rate)
    # generate text
    print(rnn_model.generate_text(char="H", length=500, top_k=5))
# %%
