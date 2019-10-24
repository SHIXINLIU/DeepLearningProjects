import tensorflow as tf
import numpy as np
from preprocess import get_data
import time

class Model(tf.keras.Model):
    def __init__(self, vocab_size):
        """
        The Model class predicts the next words in a sequence,
        Feel free to initialize any variables that you find necessary in the constructor.
        :param vocab_size: The number of unique words in the data
        """
        super(Model, self).__init__()

        # TODO: initialize vocab_size, embedding_size
        self.vocab_size = vocab_size
        self.embedding_size = 256
        self.batch_size = 1000
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

        # TODO: initialize embeddings and forward pass weights (weights, biases)
        self.E = tf.Variable(tf.random.truncated_normal(shape=[self.vocab_size, self.embedding_size], mean=0, stddev=0.1))
        self.W = tf.Variable(tf.random.truncated_normal(shape=[self.embedding_size * 2, self.vocab_size], mean=0, stddev=0.1))
        self.b = tf.Variable(tf.random.truncated_normal(shape=[1, self.vocab_size], mean=0, stddev=0.1))

    def call(self, inputs):
        """
        - You must use an embedding layer as the first layer of your network (i.e. tf.nn.embedding_lookup)
        :param inputs: word ids of shape (batch_size, 2)
        :return: probs: The batch element probabilities as a tensor of shape (batch_size, vocab_size)
        """
        # embedding = tf.nn.embedding_lookup(self.E, inputs)
        # embedding = tf.concat([embedding[:, 0, :], embedding[:, 1, :]], 1)
        embedding1 = tf.nn.embedding_lookup(self.E, inputs[:, 0])
        embedding2 = tf.nn.embedding_lookup(self.E, inputs[:, 1])
        embedding = tf.concat([embedding1, embedding2], 1)
        probs = tf.nn.softmax(tf.nn.relu(tf.matmul(embedding, self.W) + self.b))

        return probs

    def loss_function(self, probs, labels):
        """
        Calculates average cross entropy sequence to sequence loss of the prediction
        :param prbs: a matrix of shape (batch_size, vocab_size)
        :return: the loss of the model as a tensor of size 1
        """
        return tf.keras.losses.sparse_categorical_crossentropy(labels, probs)

def train(model, train_inputs, train_labels):
    """
    Runs through one epoch - all training examples. 
    You should already shape the train inputs into groups of two words.
    Remember to shuffle your inputs and labels - ensure that they are shuffled in the same order. 
    Also you should batch your input and labels here.
    :param model: the initilized model to use for forward and backward pass
    :param train_inputs: train inputs (all inputs for training) of shape (num_inputs,2)
    :param train_labels: train labels (all labels for training) of shape (num_inputs,)
    :return: None
    """
    print('Train starts: \n')
    indices = tf.range(0, train_inputs.shape[0])
    indices = tf.random.shuffle(indices)
    train_inputs = tf.gather(train_inputs, indices)
    train_labels = tf.gather(train_labels, indices)

    N = train_inputs.shape[0] // model.batch_size
    for batch in range(N):
        start = batch * model.batch_size
        end = (batch + 1) * model.batch_size
        if (batch + 1) * model.batch_size > train_inputs.shape[0]:
            end = train_inputs.shape[0]
        inputs = train_inputs[start: end]
        labels = train_labels[start: end]

        with tf.GradientTape() as tape:
            probs = model.call(inputs)
            loss = model.loss_function(probs, labels)

        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        # if batch % 100 == 0:
        #     print('Batch {} starts:'.format(batch))
        print('\r', "training processing : {} %".format((batch + 1) * 100 // N), end='')


def test(model, test_inputs, test_labels):
    """
    Runs through all test examples. You should take the test input and shape them into groups of two words.
    :param model: the trained model to use for prediction
    :param test_input: train inputs (all inputs for testing) of shape (num_inputs,2)
    :param test_input: train labels (all labels for testing) of shape (num_inputs,)
    :returns: the perplexity of the test set
    """
    print('\nTest starts:')
    probs = model.call(test_inputs)
    avg_loss = tf.reduce_mean(model.loss_function(probs, test_labels))
    return np.exp(avg_loss)


def generate_sentence(word1, word2, length, vocab, model):
    """
    Given initial 2 words, print out predicted sentence of target length.
    :param word1: string, first word
    :param word2: string, second word
    :param length: int, desired sentence length
    :param vocab: dictionary, word to id mapping
    :param model: trained trigram model
    """
    reverse_vocab = {idx: word for word, idx in vocab.items()}
    output_string = np.zeros((1, length), dtype=np.int)
    output_string[:, 0: 2] = vocab[word1], vocab[word2]

    for end in range(2, length):
        start = end - 2
        output_string[:, end] = np.argmax(model(output_string[:, start:end]), axis=1)
    text = [reverse_vocab[i] for i in list(output_string[0])]
    
    print(" ".join(text))


def main():
    # TODO: Pre-process and vectorize the data using get_data from preprocess
    train_token, test_token, vocab_dict = get_data('data/train.txt', 'data/test.txt')
    num_train = train_token.shape[0]
    num_test = test_token.shape[0]

    # TODO:  Separate your train and test data into inputs and labels
    train_inputs = np.zeros((num_train, 2), dtype=np.int32)
    train_labels = np.zeros((num_train, 1), dtype=np.int32)
    for i in range(num_train - 2):
        train_inputs[i, :] = train_token[i: i + 2]
        train_labels[i] = train_token[i + 2]
    test_inputs = np.zeros((num_test, 2), dtype=np.int32)
    test_labels = np.zeros((num_test, 1), dtype=np.int32)
    for i in range(num_test - 2):
        test_inputs[i, :] = test_token[i: i + 2]
        test_labels[i] = test_token[i + 2]

    # TODO: initialize model and tensorflow variables
    vocab_size = 7342
    model = Model(vocab_size)

    # TODO: Set-up the training step
    train(model, train_inputs, train_labels)

    # TODO: Set up the testing steps
    perplexity = test(model, test_inputs, test_labels)
    print('Perplexity = {}'.format(perplexity))
    # Print out perplexity

    word1 = 'I'
    word2 = 'like'
    length = 10
    generate_sentence(word1, word2, length, vocab_dict, model)
    
if __name__ == '__main__':
    main()
