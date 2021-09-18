import os
import re
from collections import Counter
import numpy as np
import pandas as pd


def process_data(file):
    # input: txt file
    # output: a list of words 
    with open(file, 'r') as f:
        read_data = f.read()
        read_data = read_data.lower()
        str_pattern = r'\w+'  # numbers also count
        matches = re.findall(str_pattern, read_data)

    return matches


def get_count(wordlist):
    # input: a list of words
    # output: a dictionary of words with counting values
    word_dict = {}
    for i in wordlist:
        word_dict[i] = word_dict.get(i, 0) + 1 # you can use Counter(wordlist)

    return word_dict


def get_probs(word_count_dict):
    '''
    input: word count dictionary
    output: A dictionary where keys are the words and the values are the 
            probability that a word will occur. 
    '''
    probs = {v:k/sum(list(word_count_dict.values())) 
                            for v, k in word_count_dict.items()}

    return probs 


def delete_letter(word, verbose=False):
    '''
    Input: a string
    Output: a list of all possible strings obtained by deleting 1 character 
            the string
    '''
    splits = [(word[:i], word[i:]) for i in range(len(word)+1)]
    deletes = [(l+r[1:]) for l, r in splits if r]

    if verbose: print(
        f"input word: {word}\n"
        f"splits: {splits}\n"
        f"deletes: {deletes}")

    return deletes


if __name__ == "__main__":
    path = os.getcwd()+'/assignments/autocorrect'
    os.chdir(path)
    word_list = process_data('./data/shakespeare.txt')
    vocab = set(word_list)
    print(f'The first ten words in the list are:\n {word_list[:10]}')
    print(f'There are {len(vocab)} unique words in the vocabulary')
    word_count_dict = get_count(word_list)
    print(f'There are {len(word_count_dict)} key value pairs')
    print(f"The count for the word 'thee' is {word_count_dict.get('thee', 0)}")
    probs = get_probs(word_count_dict)
    print(f"Length of probs is {len(probs)}")
    print(f"P('thee') is {probs['thee']:.4f}")
    delete_letter('cans', verbose=True)
    print(
        f"Number of outputs of delete_letter('at') "
        f"is {len(delete_letter('at'))}")