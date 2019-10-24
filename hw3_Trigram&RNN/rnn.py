import tensorflow as tf
import numpy as np
import time
from preprocess import get_data


class Model(tf.keras.Model):
    def __init__(self, vocab_size):
        """
        The Model class predicts the next words in a sequence.
        Feel free to initialize any variables that you find necessary in the constructor.
        :param vocab_size: The number of unique words in the data
        """
        super(Model, self).__init__()

        # TODO: initialize vocab_size, emnbedding_size
        self.vocab_size = vocab_size
        self.window_size = 20
        self.embedding_size = 300
        self.batch_size = 300
        self.rnn_size = 256  # ouptput size of rnn
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

        # TODO: initialize embeddings and forward pass weights (weights, biases)
        # Note: You can now use tf.keras.layers!
        # use tf.keras.layers. Dense for feed forward layers: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense
        # and use tf.keras.layers.GRU or tf.keras.layers.LSTM for your RNN
        self.E = tf.Variable(tf.random.truncated_normal(shape=[self.vocab_size, self.embedding_size], mean=0, stddev=0.1))
        self.lstm = tf.keras.layers.LSTM(self.rnn_size, return_sequences=True, return_state=True)
        self.dense = tf.keras.layers.Dense(self.vocab_size, activation='softmax')

    def call(self, inputs, initial_state=None):
        """
        - You must use an embedding layer as the first layer of your network (i.e. tf.nn.embedding_lookup)
        - You must use an LSTM or GRU as the next layer.
        :param inputs: word ids of shape (batch_size, window_size)
        :param initial_state: 2-d array of shape (batch_size, rnn_size) as a tensor
        :return: the batch element probabilities as a tensor, the final_state(s) of the rnn
        -Note 1: If you use an LSTM, the final_state will be the last two outputs of calling the rnn.
        If you use a GRU, it will just be the second output.
        -Note 2: You only need to use the initial state during generation. During training and testing it can be None.
        """
        embedding = tf.nn.embedding_lookup(self.E, inputs)  # (batch_size, embedding_size, window_size)
        output, state1, state2 = self.lstm(embedding, initial_state=initial_state)
        dense = self.dense(output)

        return dense, (state1, state2)

    def loss(self, probs, labels):
        """
        Calculates average cross entropy sequence to sequence loss of the prediction
        :param logits: a matrix of shape (batch_size, window_size, vocab_size) as a tensor
        :param labels: matrix of shape (batch_size, window_size) containing the labels
        :return: the loss of the model as a tensor of size 1
        """
        #We recommend using tf.keras.losses.sparse_categorical_crossentropy
        #https://www.tensorflow.org/api_docs/python/tf/keras/losses/sparse_categorical_crossentropy
        return tf.keras.losses.sparse_categorical_crossentropy(labels, probs)


def train(model, train_inputs, train_labels):
    """
    Runs through one epoch - all training examples.
    :param model: the initilized model to use for forward and backward pass
    :param train_inputs: train inputs (all inputs for training) of shape (num_inputs, window_size)
    :param train_labels: train labels (all labels for training) of shape (num_labels, window_size)
    :return: None
    """
    print('Train starts: \n')
    initial_state = None
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
            probs, _ = model.call(inputs, initial_state)
            loss = model.loss(probs, labels)

        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        # if batch % 100 == 0:
        #     print('Batch {} starts:'.format(batch))
        print('\r', "training processing : {} %".format((batch + 1) * 100 // N), end='')


def test(model, test_inputs, test_labels):
    """
    Runs through one epoch - all testing examples
    :param model: the trained model to use for prediction
    :param test_inputs: train inputs (all inputs for testing) of shape (num_inputs, window_size)
    :param test_labels: train labels (all labels for testing) of shape (num_labels, window_size)
    :returns: perplexity of the test set
    Note: perplexity is exp(total_loss/number of predictions)
    """
    print('\nTest starts:')
    probs, _ = model.call(test_inputs)
    avg_loss = tf.reduce_mean(model.loss(probs, test_labels))
    return np.exp(avg_loss)


def generate_sentence(word1, length, vocab, model):
    """
    Takes a model, vocab, selects from the most likely next word from the model's distribution
    This is only for your own exploration. What do the sequences your RNN generates look like?
    :param word1: string, the first word
    :param length: int, desired length of the sentence
    :param vocab: dictionary, word to id mapping
    :param model: trained RNN model
    :return: None
    """
    reverse_vocab = {idx:word for word, idx in vocab.items()}
    previous_state = None

    first_string = word1
    first_word_index = vocab[word1]
    next_input = [[first_word_index]]
    text = [first_string]

    for i in range(length):
        logits,previous_state = model.call(next_input,previous_state)
        out_index = np.argmax(np.array(logits[0][0]))

        text.append(reverse_vocab[out_index])
        next_input = [[out_index]]

    print(" ".join(text))


def main():
    # TODO: Pre-process and vectorize the data
    train_token, test_token, vocab_dict = get_data('data/train.txt', 'data/test.txt')
    num_train = train_token.shape[0]
    num_test = test_token.shape[0]
    # HINT: Please note that you are predicting the next word at each timestep, so you want to remove the last element
    # from train_x and test_x. You also need to drop the first element from train_y and test_y.
    # If you don't do this, you will see very, very small perplexities.

    # TODO: initialize model and tensorflow variables
    vocab_size = 7342
    model = Model(vocab_size)

    # TODO:  Separate your train and test data into inputs and labels
    num_train = (num_train - 1) // model.window_size
    num_test = (num_test - 1) // model.window_size
    train_inputs = np.zeros((num_train, model.window_size), dtype=np.int32)
    train_labels = np.zeros((num_train, model.window_size), dtype=np.int32)
    for i in range(num_train):
        train_inputs[i] = train_token[i * model.window_size: (i + 1) * model.window_size]
        train_labels[i] = train_token[i * model.window_size + 1: (i + 1) * model.window_size + 1]
    test_inputs = np.zeros((num_test, model.window_size), dtype=np.int32)
    test_labels = np.zeros((num_test, model.window_size), dtype=np.int32)
    for i in range(num_test):
        test_inputs[i] = test_token[i * model.window_size: (i + 1) * model.window_size]
        test_labels[i] = test_token[i * model.window_size + 1: (i + 1) * model.window_size + 1]

    # TODO: Set-up the training step
    train(model, train_inputs, train_labels)

    # TODO: Set up the testing steps
    Perplexity = test(model, test_inputs, test_labels)
    print('Perplexity = {}'.format(Perplexity))

    generate_sentence('I', 10, vocab_dict, model)
    
if __name__ == '__main__':
    main()
