import tensorflow as tf
import numpy as np
from functools import reduce

def get_data(train_file, test_file):
    """
    Read and parse the train and test file line by line, then tokenize the sentences to build the train and test data separately.
    Create a vocabulary dictionary that maps all the unique tokens from your train and test data as keys to a unique integer value.
    Then vectorize your train and test data based on your vocabulary dictionary.

    :param train_file: Path to the training file.
    :param test_file: Path to the test file.
    :return: Tuple of train (1-d list or array with training words in vectorized/id form), test (1-d list or array with testing words in vectorized/id form), vocabulary (Dict containg word->idx mapping)
    """

    # TODO: load and concatenate training data from training file.
    with open(train_file, 'r') as f:
        train_data = f.read().split()

    # TODO: load and concatenate testing data from testing file.
    with open(test_file, 'r') as f:
        test_data = f.read().split()

    # generate vocab dictionary
    vocab_set = set(train_data + test_data)
    vocab_dict = {j: i for i, j in enumerate(vocab_set)}

    # TODO: read in and tokenize training data
    train_token = []
    for s in train_data:
        train_token.append(vocab_dict.get(s))
    train_token = np.array(train_token)
    train_token = train_token.astype(dtype=np.int32)

    # TODO: read in and tokenize testing data
    test_token = []
    for s in test_data:
        test_token.append(vocab_dict.get(s))
    test_token = np.array(test_token)
    test_token = test_token.astype(dtype=np.int32)

    # TODO: return tuple of training tokens, testing tokens, and the vocab dictionary.

    return train_token, test_token, vocab_dict
