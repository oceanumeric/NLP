# given a list of words [sliced with a fixed window size]
# predict the center word 
import os
import nltk
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from scipy import linalg
from nltk.tokenize import word_tokenize 
from collections import Counter, defaultdict


nltk.data.path.append('.')  # save tokenizer data into the current path


class CBOW:
    """
    Continuous Bag of Words Model (word2vec) step by step:
        - read the text and tokenize it
        - map words to indices and indices to words
        - create a dictonary for each vocabuarly with index 
        - initialize the model
    """
    
    def __init__(self, text_file_path):
        with open(text_file_path) as f:
            self.data = f.read()
        self.token = self._tokenize(self.data)
        # a list of frequency tuple
        self.freq_dist = nltk.FreqDist(word for word in self.token)
        self.word2ind, self.ind2word = self._get_dict(self.token)
        
        print(len(self.word2ind))
            
    def _tokenize(self, data):
        data = re.sub(r'[,!?;-]', '.', data)
        data = nltk.word_tokenize(data)
        data = [ch.lower() for ch in data if ch.isalpha() or ch == '.']
        return data
    
    def _get_dict(self, token):
        words = sorted(list(set(token)))
        idx = 0
        word2ind = {}
        ind2word = {}
        for k in words:
            word2ind[k] = idx
            ind2word[idx] = k
            idx += 1
        return word2ind, ind2word
    
    def _get_idx(self, context_words, word2ind):
        idx = []
        for word in context_words:
            idx += [word2ind[word]]
        return idx
    
    def _pack_idx_with_frequency(self, context_words, word2ind):
        freq_dict = defaultdict(int)
        for word in context_words:
            freq_dict[word] += 1
        idxs = self._get_idx(context_words, word2ind)
        packed = []
        for i in range(len(idxs)):
            idx = idxs[i]
            freq = freq_dict[context_words[i]]
            packed.append((idx, freq))
        
        return packed
    
    def _get_vectors(self, center):
        i = center
        v = len(self.word2ind)
        while True:
            y = np.zeros(v)
            x = np.zeros(v)
            center_word = self.token[i]
            y[self.word2ind[center_word]] = 1 
            # context window size = 4 (without counting center word)
            context_words = self.token[(i-center):i] + self.token[(i+1):(i+center+1)]
            num_ctx_words = len(context_words)
            for idx, freq in self._pack_idx_with_frequency(context_words, self.word2ind):
                x[idx] = freq/num_ctx_words
            yield x, y 
            
            i += 1 
            
            if i >= len(self.token):
                print('i is being set to 0')
                i = 0 
            
    def _get_batches(self, center, batch_size):
        '''
        Input:
            self: token, word2ind
            center: position of center word
            batch size
        Output:
            a batch of data
        '''
        batch_x = []
        batch_y = []
        for x, y in self._get_vectors(center):
            while len(batch_x) < batch_size:
                batch_x.append(x)
                batch_y.append(y)
            else:
                yield np.array(batch_x).T, np.array(batch_y).T
       
    
    def _initialize_model(self, N, random_seed=1):
        '''
        Input:
            N - dimension of hidden vector
            random seed
        Output:
            initialized weights and biases: w1, b1, w2, b2
        '''
        np.random.seed(random_seed)
        v = len(self.word2ind)
        w1 = np.random.rand(N, v)  # shape = (N, v)
        w2 = np.random.rand(v, N)  # shape = (v, N)
        b1 = np.random.rand(N, 1)
        b2 = np.random.rand(v, 1)
        
        return w1, b1, w2, b2
    
    def _softmax(self, z):
        '''
        Input:
            z: output scores from the hidden layer, (v, m) m = bach size
        Output:
            yhat: prediction 
        '''
        z = np.float128(z)
        z_exp = np.exp(z)
        y_hat = z_exp / np.sum(z_exp, axis=0) 
        return y_hat.T
    
    def _forward_prop(self, x, w1, b1, w2, b2):
        '''
        Input:
            x - average one-hot vector from the context shape = (v, 1)
            w1, b1, w2, b2 - parameters 
        Output:
            hidden layer h = w1 x + b1, z = w2 Relu(h) + b2
        '''
        h = w1 @ x + b1  # w1 shape = (n, v)
        h = np.maximum(0, h)
        z = w2 @ h + b2  # w2 shape = (v, n)
        
        return z.T, h.T
    
    def _compute_cost(self, z, y, yhat, c, batch_size):
        '''
        Input:
            z: shape = (v, n)
        '''
        # element-wise multiplication 
        z_hat = logsumexp(z, axis=1, keepdims=True)
        cost = (-np.sum(y*np.log(yhat)) + np.sum(2.0*c*z_hat)) / batch_size
        cost = np.squeeze(cost)  # remove dimension (shape) indices 
        
        return cost
        
    def _back_prop(self, x, yhat, y, h, w1, b1, w2, b2, batch_size, m):
        '''
        Input:
            as it states
            batch_size: the training batch 
            m: number of context words
        Output:
            gradients 
            grad_w1, grad_w2, grad_b1, grad_b2
        '''
        w2y = np.dot(w2.T, (yhat - y)) 
        
        w2yrelu = np.maximum(0, w2y)
        grad_w1 = (1/(batch_size*m))*np.dot(w2yrelu, x.T)  # coefficient of w1 = X 
        grad_b1 = (1/batch_size)*np.sum(w2yrelu, axis=1, keepdims=True)
        grad_w2 = (1/(batch_size*m))*np.dot(yhat-y, h)
        grad_b2 = (1/(batch_size*m))*np.sum((yhat-y), axis=1, keepdims=True)
        
        return grad_w1, grad_b1, grad_w2, grad_b2
    
    def gradient_descent(self, N, num_iters, alpha=0.03):
        '''
        Input:
            text data: self.token, self.word2ind
            N: dimension of hidden layzer 
            num_iters
            learning rate: alpha
        Output:
            updated gradients: w1, b1, w2 b2
        '''
        w1, b1, w2, b2 = self._initialize_model(N)
        center = 2
        m = 2*center  # context window [i am happy(center) become I]
        batch_size = 128
        iters = 0
        for x, y in self._get_batches(center, batch_size):
            z, h = self._forward_prop(x, w1, b1, w2, b2)
            yhat = self._softmax(z)
            cost = self._compute_cost(z, y, yhat, center, batch_size)
            
            if ((iters + 1) % 2 == 0):
                print(f"iterations: {iters + 1} cost: {cost:.6f}")
            # get gradients 
            grad_w1, grad_b1, grad_w2, grad_b2 = self._back_prop(x, yhat, y, h,
                                                                 w1, b1, w2, b2,
                                                                 batch_size, m)
            w1 = w1 - alpha * grad_w1
            b1 = b1 - alpha * grad_b1
            w2 = w2 - alpha * grad_w2
            b2 = b2 - alpha * grad_b2
            
            iters += 1
            if iters == num_iters:
                break
            if iters % 100 == 0:
                # update learnign rate
                alpha *= 0.66
                
        return w1, b1, w2, b2
    

def compute_pca(data, n_components=2):
    """
    Input: 
        data: of dimension (m,n) where each row corresponds to a word vector
        n_components: Number of components you want to keep.
    Output: 
        X_reduced: data transformed in 2 dims/columns + regenerated original data
    pass in: data as 2D NumPy array
    """

    m, n = data.shape

    ### START CODE HERE ###
    # mean center the data
    data -= data.mean(axis=0)
    # calculate the covariance matrix
    R = np.cov(data, rowvar=False)
    # calculate eigenvectors & eigenvalues of the covariance matrix
    # use 'eigh' rather than 'eig' since R is symmetric,
    # the performance gain is substantial
    evals, evecs = linalg.eigh(R)
    # sort eigenvalue in decreasing order
    # this returns the corresponding indices of evals and evecs
    idx = np.argsort(evals)[::-1]

    evecs = evecs[:, idx]
    # sort eigenvectors according to same index
    evals = evals[idx]
    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    evecs = evecs[:, :n_components]
    ### END CODE HERE ###
    return np.dot(evecs.T, data.T).T

    
if __name__ == "__main__":
    print(os.getcwd())
    print(defaultdict(int))
    word2vec = CBOW('./word_embeddings/shakespeare.txt')
    # train the model N = 50, iteration = 150
    w1, b1, w2, b2 = word2vec.gradient_descent(50, 15)
    print(w1.shape, b1.shape, w2.shape, b2.shape)
    words = ['king', 'queen','lord','man', 'woman','dog','horse',
         'rich','happy','sad']
    embs = (w1.T+w2)/2
    idx = [word2vec.word2ind[word] for word in words]
    words_vector = embs[idx, :]
    print(words_vector.shape)
    
     
    
    





