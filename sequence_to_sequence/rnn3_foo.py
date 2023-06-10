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
            h_t = torch.tanh(x_t @ self.Wxh + h_t @ self.Whh+ self.bh)
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

        return logits, h_t.squeeze(1)  # 
    

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

        with open(data_path, 'r') as f:
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
            yield xyield, yyield
    

def predict(model, char, device, h=None, top_k=5):
        ''' Given a character & hidden state, predict the next character.
            Returns the predicted character and the hidden state.
        '''
        
        # tensor inputs
        x = torch.tensor([[model.char_to_int[char]]]).to(device)
        # put into tensor
        inputs = F.one_hot(x, model.vocab_size).float()
        
        with torch.no_grad():
            # get the output of the model
            out, h = model(inputs, h)

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


if __name__ == "__main__":
    print("Hello World")
    print(os.getcwd())

    seq_length = 100
    batch_size = 128   # 512
    hidden_size = 512  # or 256
    epochs = 300
    learning_rate = 0.001

    rnn_model = SimpleRNN(data_path="data/sonnets.txt", batch_size=batch_size,
                            seq_length=seq_length, hidden_size=hidden_size)
    # foox, fooy = next(rnn_model.create_batches(10, 50))
    # print(foox.shape, fooy.shape)
    # print(foox[:10, :10])
    # print(fooy[:10, :10])

    # print out number of parameters
    print(f"Number of parameters is {sum(p.numel() for p in rnn_model.parameters())}")

    # push to GPU
    rnn_model.to(device)

    # train the model
    loss_list = rnn_model.train_model(epochs=epochs, lr=learning_rate)

     # print(model.generate_text(char="A", length=100, top_k=5))
    print(sample(rnn_model, 1000, device, prime='A', top_k=5))
# %%
