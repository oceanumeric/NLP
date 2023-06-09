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


class SimpleRNN(nn.Module):
    """
    We will use a simple RNN to predict the next character in a sequence of characters.
    The input is a sequence of characters, and the output is a sequence of characters. \\
    it will have 2 layers: \\
        1. a simple RNN layer \\
        2. a fully connected layer
    Reference: https://youtu.be/ySEx_Bqxvvo
    """

    def __init__(self, hidden_size, drop_prob=0.5):
        super(SimpleRNN, self).__init__()

        # load the data
        self.load_data("./data/input.txt")
        self.process_data()
        self.create_batches(12, 126)

        # we will not use embedding layer
        # as we will use one-hot encoding
        self.hidden_size = hidden_size
        self.drop_prob = drop_prob

        # initialize the parameters
        self.W_xh = nn.Parameter(torch.randn(1, hidden_size))



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
        self.X = self.X.view(batch_size*self.n_batches, seq_length)
        self.Y = self.Y.view(batch_size*self.n_batches, seq_length)
        print(f"The shape of X is {self.X.shape}; and the shape of Y is {self.Y.shape}")

    
    def load_batch(self):
        # a batch generator to generate X and Y
        for i in range(0, self.X.shape[0], self.batch_size):
            x = self.X[i:i+self.batch_size, :]
            y = self.Y[i:i+self.batch_size, :]
            # print(f"The shape of x is {x.shape}; and the shape of y is {y.shape}")
            # print(f"The first 10 characters in x are {x[0, :10]}")
            # print(f"The first 10 characters in y are {y[0, :10]}")
            yield x, y

        



if __name__ == "__main__":
    print("Hello World!")
    print(os.getcwd())
    foo = SimpleRNN()
    foo.load_data("./data/input.txt")
    foo.process_data()
    foo.create_batches(12, 126)
    # print(next(foo.load_batch())
# %%
