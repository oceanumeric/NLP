import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import random
import matplotlib.pyplot as plt


# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
embedding_dim = 384
num_heads = 6 # how many heads to use in the multi-head attention
block_layers = 6 # how many transformer blocks to use
drop_out = 0.2 # dropout rate


torch.manual_seed(1337)
print(f'Using device {device}')



# read the dataset
with open('./data/input.txt', 'r') as f:
    text = f.read()


chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a dictionary that maps integers to characters and vice versa
int2char = dict(enumerate(chars))
char2int = {ch: ii for ii, ch in int2char.items()}

# create encode and decode functions
encode = lambda text: [char2int[ch] for ch in text]
decode = lambda int_arr: ''.join([int2char[ii] for ii in int_arr])


# encode the whole text
text_data = torch.tensor(encode(text), dtype=torch.long)


# split the dataset into train and test sets
train_n = int(text_data.shape[0] * 0.9)
train_data = text_data[:train_n]
test_data = text_data[train_n:]



def get_batch(split):
    # generate random starting indices for the batch data
    data = train_data if split == 'train' else test_data
    # get the starting indices for the batch data
    starts = torch.randint(high=data.shape[0] - block_size, size=(batch_size,))
    # get the batch data
    batch_x = [data[start:start+block_size] for start in starts]
    batch_y = [data[start+1:start+block_size+1] for start in starts]
    # convert the list to tensors
    batch_x, batch_y = torch.stack(batch_x), torch.stack(batch_y)
    # push the data to the device
    batch_x, batch_y = batch_x.to(device), batch_y.to(device)

    return batch_x, batch_y


# attention mechanism
class HeadAttention(nn.Module):

    """One-head of self-attention mechanism"""

    def __init__(self, head_size):
        super().__init__()
        # create query, key, value matrices
        self.query = nn.Linear(embedding_dim, head_size, bias=False)
        self.key = nn.Linear(embedding_dim, head_size, bias=False)
        self.value = nn.Linear(embedding_dim, head_size, bias=False)
        # create a buffer for the triangular matrix
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        # add dropout
        self.drop_out = nn.Dropout(drop_out)

    def forward(self, x):
        B, T, C = x.shape
        # get the query, key, value matrices
        q = self.query(x) #(B, T, head_size)
        k = self.key(x) #(B, T, head_size)
        v = self.value(x) #(B, T, head_size)
        # compute the scaled dot product
        # (B, T, head_size) x (B, head_size, T) = (B, T, T)
        scaled_dot_product = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(C)
        # apply the mask
        scaled_dot_product = scaled_dot_product.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        # apply softmax
        attention = F.softmax(scaled_dot_product, dim=-1) #(B, T, T)
        # apply dropout
        attention = self.drop_out(attention)
        # apply the attention to the value
        # (B, T, T) x (B, T, head_size) = (B, T, head_size)
        out = torch.bmm(attention, v) #(B, T, head_size)

        return out
    

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism"""

    def __init__(self, num_heads, head_size):

        super().__init__()
        # create the heads
        self.heads = nn.ModuleList([HeadAttention(head_size) for _ in range(num_heads)])
        # create the output projection
        self.proj = nn.Linear(num_heads * head_size, embedding_dim)
        # add drop out
        self.drop_out = nn.Dropout(drop_out)
        
    def forward(self, x):
        # concatenate the heads
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        # project the output
        out = self.proj(out)
        return out
    

class FeedForward(nn.Module):

    def __init__(self, embedding_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.ReLU(),
            nn.Linear(4 * embedding_dim, embedding_dim),  # project back to embedding dim
            ## add dropout
            nn.Dropout(drop_out)
        )

    def forward(self, x):
        return self.net(x)
    

class Block(nn.Module):
    """Transformer block: communication followed by feed-forward"""

    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        head_size = embedding_dim // num_heads
        self.sa = MultiHeadAttention(num_heads, head_size)
        self.ff = FeedForward(embedding_dim)
        # add layer normalization
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        # apply the self-attention
        x = x + self.sa(self.ln1(x))
        # apply the feed-forward
        x = x + self.ff(self.ln2(x))
        return x



class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()

        # embedding layer
        # it is a matrix of size vocab_size x vocab_size
        # which serves as a lookup table for the token embeddings
        # what is lookup table?
        # it is a table that maps integers to embeddings
        # right now, it is initialized randomly
        # but it will be trained later and learned from the data
        # here the vector size is the same as the vocab size
        self.token_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # postional embeddings
        # it is a matrix of size block_size x embedding_dim
        # here block size is the number of tokens in the context == 8 
        self.position_embeddings = nn.Embedding(block_size, embedding_dim)
        # add blocks - 3 blocks
        self.blocks = nn.Sequential(
            *[Block(embedding_dim, num_heads=num_heads) for _ in range(block_layers)],
            nn.LayerNorm(embedding_dim)
            )
        self.lm_head = nn.Linear(embedding_dim, vocab_size)

    
    def forward(self, idx, target=None):

        B, T = idx.shape

        # get the token embeddings  (batch_size, block_size, channel_size)
        # block_size is the number of tokens in the context
        # which is also called the time steps, that's why it is T
        token_embeddings = self.token_embeddings(idx) #(B, T, C)
        position_embeddings = self.position_embeddings(torch.arange(T).to(device)) #(T, C)
        # add the positional embeddings to the token embeddings
        # the shape of the positional embeddings will be broadcasted to (B, T, C)
        # because the first dimension is 1 and torch will automatically broadcast it
        x = token_embeddings + position_embeddings #(B, T, C)
        # apply the attention heads
        x = self.blocks(x) #(B, T, C)
        logits = self.lm_head(x) #(B, T, vocab_size)

        if target is None:
            loss = None
        else:
            # reshape the embeddings to (batch_size * block_size, channel_size)
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            target = target.view(B * T)
            # compute the loss
            loss = F.cross_entropy(logits, target)
    
        return logits, loss 
    
    def generate(self, idx, max_new_tokens):
        # idx is (batch_size, block_size)  which is (B, T)
        # which means the context for each batch
        for _ in range(max_new_tokens):
            # crop the idx
            # we only need the last block_size tokens
            idx_crop = idx[:, -block_size:]
            # get the predictions
            # self(idx, target) is the same as self.forward(idx, target)
            logits, loss = self(idx_crop)
            # get the logit for the last token in the context
            # which is the token we want to predict
            # here logits is (batch_size, block_size, channel_size)
            # beacause loss=None, we did not reshape the logits
            logits = logits[:, -1, :]
            # the shape now is (batch_size, channel_size)
            # dim=-1 means the last dimension, which is the channel_size
            # we want to normalize the logits to get the probabilities
            # in the dimension of channel_size
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append the new token to the context
            idx = torch.cat([idx, idx_next], dim=1)

        return idx
    
m1 = BigramLanguageModel(vocab_size).to(device)

optimizer = torch.optim.Adam(m1.parameters(), lr=1e-3)


# function to evaluate the loss
@torch.no_grad()
def est_loss():
    out = {}
    # set the model to evaluation mode
    m1.eval()
    for split in ['train', 'test']:
        # eval_iters means how many iterations we will evaluate
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            logits, loss = m1(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean()
    # set the model back to training mode
    m1.train()
    return out


for i in range(max_iters):

    # evaluate the loss every eval_interval iterations

    if i % eval_interval == 0:
        losses = est_loss()
        print(f'iteration {i}, train loss {losses["train"]:.4f}, test loss {losses["test"]:.4f}')

    xb, yb = get_batch('train')

    # forward pass
    logits, loss = m1(xb, yb)
    # clear the gradients
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    # update the parameters
    optimizer.step()

# print out the final loss
print(loss.item())
    
# generate some text
context = torch.zeros((1, 1), dtype=torch.long, device=device)
foo = m1.generate(context, max_new_tokens=276)
print(decode(foo[0].tolist()))



