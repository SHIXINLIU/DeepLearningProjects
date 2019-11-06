import numpy as np
import tensorflow as tf

class RNN_Seq2Seq(tf.keras.Model):
	def __init__(self, french_window_size, french_vocab_size, english_window_size, english_vocab_size):
		###### DO NOT CHANGE ######
		super(RNN_Seq2Seq, self).__init__()
		self.french_vocab_size = french_vocab_size # The size of the french vocab
		self.english_vocab_size = english_vocab_size # The size of the english vocab
		self.french_window_size = french_window_size # The french window size
		self.english_window_size = english_window_size # The english window size
		###### DO NOT CHANGE ######
		# TODO:
		# 1) Define any hyper parameters
		# Define batch size and optimizer/learning rate
		self.batch_size = 100
		self.embedding_size = 64
		self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
		self.rnn_size = 80

		# 2) Define embeddings, encoder, decoder, and feed forward layers
		self.EE = tf.Variable(
			tf.random.truncated_normal(shape=[self.english_vocab_size, self.embedding_size], mean=0, stddev=0.01))
		self.EF = tf.Variable(
			tf.random.truncated_normal(shape=[self.french_vocab_size, self.embedding_size], mean=0, stddev=0.01))

		self.encoder = tf.keras.layers.LSTM(self.rnn_size, return_sequences=True, return_state=True)
		self.decoder = tf.keras.layers.LSTM(self.rnn_size, return_sequences=True, return_state=True)
		self.dense = tf.keras.layers.Dense(self.english_vocab_size, activation='softmax')

	@tf.function
	def call(self, encoder_input, decoder_input):
		"""
		:param encoder_input: batched ids corresponding to french sentences
		:param decoder_input: batched ids corresponding to english sentences
		:return prbs: The 3d probabilities as a tensor, (batch_size x window_size x english_vocab_size)
		"""
		# TODO:
		# 1) Pass your french sentence embeddings to your encoder
		fre_embed = tf.nn.embedding_lookup(self.EF, encoder_input)
		encoder_out, state1, state2 = self.encoder(fre_embed, initial_state=None)

		# 2) Pass your english sentence embeddings, and final state of your encoder, to your decoder
		eng_embed = tf.nn.embedding_lookup(self.EE, decoder_input)
		decoder_out, state3, state4 = self.decoder(eng_embed, initial_state=(state1, state2))

		# 3) Apply dense layer(s) to the decoder out to generate probabilities
		prbs = self.dense(decoder_out)
		return prbs

	def accuracy_function(self, prbs, labels, mask):
		"""
		DO NOT CHANGE
		Computes the batch accuracy
		:param prbs: float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
		:param labels: integer tensor, word prediction labels [batch_size x window_size]
		:param mask: tensor that acts as a padding mask [batch_size x window_size]
		:return: scalar tensor of accuracy of the batch between 0 and 1
		"""
		decoded_symbols = tf.argmax(input=prbs, axis=2)
		accuracy = tf.reduce_mean(tf.boolean_mask(tf.cast(tf.equal(decoded_symbols, labels), dtype=tf.float32), mask))
		return accuracy

	def loss_function(self, prbs, labels, mask):
		"""
		Calculates the model cross-entropy loss after one forward pass
		:param prbs: float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
		:param labels: integer tensor, word prediction labels [batch_size x window_size]
		:param mask: tensor that acts as a padding mask [batch_size x window_size]
		:return: the loss of the model as a tensor
		"""
		return tf.boolean_mask(tf.keras.losses.sparse_categorical_crossentropy(labels, prbs), mask)
