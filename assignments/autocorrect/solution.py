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
        word_dict[i] = word_dict.get(i, 0) + 1  # you can use Counter(wordlist)

    return word_dict


def get_probs(word_count_dict):
    '''
    input: word count dictionary
    output: A dictionary where keys are the words and the values are the 
            probability that a word will occur. 
    '''
    probs = {v: k/sum(list(word_count_dict.values()))
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

    if verbose:
        print(
            f"input word: {word}\n"
            f"splits: {splits}\n"
            f"deletes: {deletes}")

    return deletes


def switch_letter(word, verbose=False):
    '''
    Input: a string
    Output: a list of possible strings by switching adjacent letters
    '''
    splits = [(word[:i], word[i:]) for i in range(len(word)+1)]
    switch = [(l+r[1]+r[0]+r[2:]) for l, r in splits if len(r) > 1]

    if verbose:
        print(
            f"Input word = {word} \n"
            f"split_l = {splits} \nswitch_l = {switch}")

    return switch


def replace_letter(word, verbose=False):
    '''
    Input: a string
    Output: a list of possible strings by replacing one letter
    '''
    splits = []
    replace = []
    letters = 'abcdefghijklmnopqrstuvwxyz'
    splits = [(word[:i], word[i:]) for i in range(len(word)+1)]
    replace = [(l+x+r[1:]) for l, r in splits if r for x in letters]
    replace_set = set(replace)
    replace_set.discard(word)
    replace = sorted(list(replace_set))

    if verbose:
        print(
            f"Input word = {word} \n"
            f"split_l = {splits} \nreplace_l {replace}")

    return replace


def insert_letter(word, verbose=False):
    splits = []
    insert = []
    letters = 'abcdefghijklmnopqrstuvwxyz'
    splits = [(word[:i], word[i:]) for i in range(len(word)+1)]
    insert = [(l+x+r) for l, r in splits for x in letters]

    if verbose:
        print(
            f"Input word = {word} \n"
            f"split_l = {splits} \nreplace_l {insert}")

    return insert


def edit_one_letter(word, allow_switches=True):
    """
    Input:
        word: the string/word for which we will generate all possible wordsthat are one edit away.
    Output:
        edit_one_set: a set of words with one possible edit. Please return a set. and not a list.
    """

    edit_one_set = []
    ### START CODE HERE ###
    rw = replace_letter(word)
    iw = insert_letter(word)
    dw = delete_letter(word)
    edit_one_set = rw + iw + dw
    if allow_switches:
        sw = switch_letter(word)
        edit_one_set += sw
    edit_one_set = set(edit_one_set)
    ### END CODE HERE ###

    return edit_one_set


def edit_two_letters(word, allow_switches=True):
    '''
    Input:
        word: the input string/word 
    Output:
        edit_two_set: a set of strings with all possible two edits
    '''

    edit_two_set = set()
    word_list = edit_one_letter(word, allow_switches)
    edit_two_set = set(word_list)
    for w in word_list:
        temp = edit_one_letter(w, allow_switches)
        edit_two_set = edit_two_set.union(set(temp))

    return edit_two_set


def get_corrections(word, probs, vocab, n=2, verbose=False):
    '''
    Input: 
        word: a user entered string to check for suggestions
        probs: a dictionary that maps each word to its probability in the corpus
        vocab: a set containing all the vocabulary
        n: number of possible word corrections you want returned in the dictionary
    Output: 
        n_best: a list of tuples with the most probable n corrected words and their probabilities.
    '''
    suggestions = []
    n_best = []

    ### START CODE HERE ###
    edit_one = edit_one_letter(word)
    edit_two = edit_two_letters(word)
    suggest_words = word if word in vocab else None or vocab.intersection(
        edit_one) or vocab.intersection(edit_two)
    suggestions = suggest_words
    best_words = {}
    if suggest_words:
        for w in suggest_words:
            best_words[w] = probs[w]
    else:
        best_words[word] = 0
    
    c = Counter(best_words)
    n_best = c.most_common(n)

    ### END CODE HERE ###
    if verbose:
        print("entered word = ", word, "\nsuggestions = ", suggestions)

    return n_best


if __name__ == "__main__":
    path = os.getcwd()+'/assignments/autocorrect'
    os.chdir(path)
    word_list = process_data('./data/shakespeare.txt')
    vocab = set(word_list)
    word_count_dict = get_count(word_list)
    print(f"There are {len(word_count_dict)} key values pairs")
    print(f"The count for the word 'thee' is {word_count_dict.get('thee',0)}")
    print("-------------------------------------------------------------------")
    probs = get_probs(word_count_dict)
    print(f"Length of probs is {len(probs)}")
    print(f"P('thee') is {probs['thee']:.4f}")
    print("-------------------------------------------------------------------")
    my_word = 'dys' 
    tmp_corrections = get_corrections(my_word, probs, vocab, 2, verbose=True) # keep verbose=True
    for i, word_prob in enumerate(tmp_corrections):
        print(f"word {i}: {word_prob[0]}, probability {word_prob[1]:.6f}")

    # CODE REVIEW COMMENT: using "tmp_corrections" insteads of "cors". "cors" is not defined
    print(f"data type of corrections {type(tmp_corrections)}")
