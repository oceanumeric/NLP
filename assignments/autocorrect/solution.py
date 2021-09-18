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


if __name__ == "__main__":
    path = os.getcwd()+'/assignments/autocorrect'
    os.chdir(path)
    word_list = process_data('./data/shakespeare.txt')
    vocab = set(word_list)
    print(f'The first ten words in the list are:\n {word_list[:10]}')
    print(f'There are {len(vocab)} unique words in the vocabulary')