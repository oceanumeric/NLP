import os
import math
import string
import itertools
import numpy as np
import pandas as pd
from collections import defaultdict


def working_with_textfile():
    # read file
    with open('./data/WSJ_02-21.pos') as f:
        lines = f.readlines()
        
    words = [line.split('\t')[0] for line in lines]

    freq = defaultdict(int)
    
    for word in words:
        freq[word] += 1 
        
    vocab = [k for k, v in freq.items() if (v > 1 and k != '\n')]
    vocab.sort()

    return vocab
    

def assign_unk(word):
    """
    Assign tokens to unknown words
    """
    punct = set(string.punctuation)
    
    # suffixes
    noun_suffix = ["action", "age", "ance", "cy", "dom", "ee", "ence", "er", 
                   "hood", "ion", "ism", "ist", "ity", "ling", "ment", 
                   "ness", "or", "ry", "scape", "ship", "ty"]
    verb_suffix = ["ate", "ify", "ise", "ize"]
    adj_suffix = ["able", "ese", "ful", "i", "ian", "ible", "ic", "ish",
                  "ive", "less", "ly", "ous"]
    adv_suffix = ["ward", "wards", "wise"]
    
    # Loop the characters in the word, check if any is a digit
    if any(char.isdigit() for char in word):
        return "--unk_digit--"

    # Loop the characters in the word, check if any is a punctuation character
    elif any(char in punct for char in word):
        return "--unk_punct--"

    # Loop the characters in the word, check if any is an upper case character
    elif any(char.isupper() for char in word):
        return "--unk_upper--"

    # Check if word ends with any noun suffix
    elif any(word.endswith(suffix) for suffix in noun_suffix):
        return "--unk_noun--"

    # Check if word ends with any verb suffix
    elif any(word.endswith(suffix) for suffix in verb_suffix):
        return "--unk_verb--"

    # Check if word ends with any adjective suffix
    elif any(word.endswith(suffix) for suffix in adj_suffix):
        return "--unk_adj--"

    # Check if word ends with any adverb suffix
    elif any(word.endswith(suffix) for suffix in adv_suffix):
        return "--unk_adv--"
    
    # If none of the previous criteria is met, return plain unknown
    return "--unk--"
    
    
def get_word_tag(line, vocab):
    # If line is empty return placeholders for word and tag
    if not line.split():
        word = "--n--"
        tag = "--s--"
    else:
        # Split line to separate word and tag
        word, tag = line.split()
        # Check if word is not in vocabulary
        if word not in vocab: 
            # Handle unknown word
            word = assign_unk(word)
    return word, tag


def working_with_tags():
    # Define tags for Adverb, Noun and To (the preposition) , respectively
    tags = ['RB', 'NN', 'TO']
    # Define 'transition_counts' dictionary
    transition_counts = {
        ('NN', 'NN'): 16241,
        ('RB', 'RB'): 2263,
        ('TO', 'TO'): 2,
        ('NN', 'TO'): 5256,
        ('RB', 'TO'): 855,
        ('TO', 'NN'): 734,
        ('NN', 'RB'): 2431,
        ('RB', 'NN'): 358,
        ('TO', 'RB'): 200
    }
    
    # Store the number of tags in the 'num_tags' variable
    num_tags = len(tags)

    # Initialize a 3X3 numpy array with zeros
    transition_matrix = np.zeros((num_tags, num_tags))
    # sort the tags
    sorted_tags = sorted(tags)
    # fill in the matrix
    for i, j in itertools.product(range(num_tags), repeat=2):
        transition_matrix[i, j] = transition_counts.get((sorted_tags[i],
                                                         sorted_tags[j]))
    
    row_sum = transition_matrix.sum(axis=1, keepdims=True)
    transition_matrix = transition_matrix/row_sum
    
    def print_matrix(matrix):
        print(pd.DataFrame(matrix, index=sorted_tags, columns=sorted_tags))
        
    print_matrix(transition_matrix)
    
    # Copy transition matrix for for-loop example
    t_matrix_for = np.copy(transition_matrix)

    # Copy transition matrix for numpy functions example
    t_matrix_np = np.copy(transition_matrix)
    
    for i in range(num_tags):
        t_matrix_for[i, i] = t_matrix_for[i, i] + math.log(row_sum[i])
    print_matrix(t_matrix_for)
    
    # Save diagonal in a numpy array
    d = np.diag(t_matrix_np)
    
    # Reshape diagonal numpy array
    d = np.reshape(d, (3,1))
    
    # Perform the vectorized operation
    d = d + np.vectorize(math.log)(row_sum)

    # Use numpy's 'fill_diagonal' function to update the diagonal
    np.fill_diagonal(t_matrix_np, d)

    # Print the matrix
    print_matrix(t_matrix_np)
        

    
if __name__ == "__main__":
    os.chdir('./topics/hmm')
    working_with_tags()

    