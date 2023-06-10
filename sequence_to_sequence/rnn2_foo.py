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


# function for setting seed
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# set up seed globally and deterministically
set_seed(666)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

        # the first layer is a simple RNN layer
        self.rnn_layer = nn.RNN(self.vocab_size, self.hidden_size, batch_first=True)
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

        hidden_output, h = self.rnn_layer(x_one_hot, h)

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
        return torch.zeros(1, batch_size, self.hidden_size)

    def train_model(self, n_epochs, lr=0.001):
        # initialize the optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

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

                # retain the graph if you want to use it later
                # loss.retain_grad()

                # backward pass
                loss.backward()

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
        self.chars = sorted(set(self.text))
        self.vocab_size = len(self.chars)
        print("Vocabulary size: {}".format(self.vocab_size))

        # create a dictionary to map the characters to integers and vice versa
        self.char_to_int = {ch: i for i, ch in enumerate(self.chars)}
        self.int_to_char = {i: ch for i, ch in enumerate(self.chars)}

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
        ydata =  torch.roll(xdata, -1)

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
        # change it to sequence
        char_seq = [self.char_to_int[c] for c in char]
        # convert the character to integer
        char = torch.tensor(char_seq, dtype=torch.long)
        # push the character to the GPU
        char = char.to(device)
        # reshape char as (1, seq_size)
        char = char.reshape(1, -1)

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
            p = F.softmax(output, dim=1).data.cpu()
            # get the top characters with highest likelihood
            p, top_ch = p.topk(top_k)
            top_ch = top_ch.numpy().squeeze()

            # select the likely next character with some element of randomness
            # for more variability
            p = p.numpy().squeeze()
            char_next = np.random.choice(top_ch, p=p / p.sum())

            char_next = self.int_to_char[char_next]

        return char_next, h

    # function to generate text
    def generate_text(self, char="a", h=None, length=100, top_k=None):

        # change to evaluation mode
        self.eval()
        # call the predict function to get the next character
        chars = [ch for ch in char]

        with torch.no_grad():
            if h is None:
                h = self.init_hidden(1)
    
            for ch in chars:
                char, h = self.predict(ch, h, top_k=top_k)
            
            chars.append(char)

            for ii in range(length):
                char, h = self.predict(chars[-1], h, top_k=top_k)
                chars.append(char)

        return "".join(chars)


def predict(model, char, device, h=None, top_k=5):
        ''' Given a character & hidden state, predict the next character.
            Returns the predicted character and the hidden state.
        '''
        
        # tensor inputs
        x = torch.tensor([[model.char_to_int[char]]]).to(device)
        # put into tensor
        
        with torch.no_grad():
            # get the output of the model
            out, h = model(x, h)

            # get the character probabilities
            # move to cpu for further processing with numpy etc. 
            p = F.softmax(out, dim=1).data.cpu()

            # get the top characters with highest likelihood
            p, top_ch = p.topk(top_k)
            top_ch = top_ch.numpy().squeeze()

            # select the likely next character with some element of randomness
            # for more variability
            p = p.numpy().squeeze()
            char = np.random.choice(top_ch, p=p/p.sum())
        
        # return the encoded value of the predicted char and the hidden state
        return model.int_to_char[char], h


def sample(model, size, device, prime='A', top_k=None):
    # method to generate new text based on a "prime"/initial sequence. 
    # Basically, the outer loop convenience function that calls the above
    # defined predict method. 
    model.eval() # eval mode
    
    # Calculate model for the initial prime characters
    chars = [ch for ch in prime]
    with torch.no_grad():
        # initialize hidden with 0 in the beginning. Set our batch size to 1 
        # as we wish to generate one sequence only. 
        h = model.init_hidden(batch_size=1)
        # put hidden state on GPU
        h = h.to(device)
        for ch in prime:
            char, h = predict(model, ch, device, h=h, top_k=top_k)

        # append the characters to the sequence
        chars.append(char)

        # Now pass in the previous/last character and get a new one
        # Repeat this process for the desired length of the sequence to be 
        # generated
        for ii in range(size):
            char, h = predict(model, chars[-1], device, h=h, top_k=top_k)
            chars.append(char)

    return ''.join(chars)
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
    batch_size = 128   # 512
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
    model.train()

    # # train the model
    loss_list = model.train_model(epochs, learning_rate)

    # # generate text
    # print(model.generate_text(char="A", length=100, top_k=5))
    print(sample(model, 1000, device, prime='A', top_k=5))


# %%
