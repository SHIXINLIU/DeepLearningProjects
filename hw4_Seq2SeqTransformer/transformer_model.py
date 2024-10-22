import numpy as np
import tensorflow as tf
import transformer_funcs as transformer

class Transformer_Seq2Seq(tf.keras.Model):
	def __init__(self, french_window_size, french_vocab_size, english_window_size, english_vocab_size):
		######vvv DO NOT CHANGE vvv##################
		super(Transformer_Seq2Seq, self).__init__()

		self.french_vocab_size = french_vocab_size 		# The size of the french vocab
		self.english_vocab_size = english_vocab_size 	# The size of the english vocab

		self.french_window_size = french_window_size 	# The french window size
		self.english_window_size = english_window_size 	# The english window size
		######^^^ DO NOT CHANGE ^^^##################
		# TODO:
		# 1) Define any hyperparameters
		self.batch_size = 100
		self.embedding_size = 128
		self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

		# 2) Define embeddings, encoder, decoder, and feed forward layers
		# Define english and french embedding layers:
		self.EE = tf.Variable(
			tf.random.truncated_normal(shape=[self.english_vocab_size, self.embedding_size], mean=0, stddev=0.01))
		self.EF = tf.Variable(
			tf.random.truncated_normal(shape=[self.french_vocab_size, self.embedding_size], mean=0, stddev=0.01))

		# Create positional encoder layers
		self.position_fre = transformer.Position_Encoding_Layer(french_window_size, self.embedding_size)
		self.position_eng = transformer.Position_Encoding_Layer(english_window_size, self.embedding_size)

		# Define encoder and decoder layers:
		self.encoder = transformer.Transformer_Block(self.embedding_size, is_decoder=False)
		self.decoder = transformer.Transformer_Block(self.embedding_size, is_decoder=True)

		# Define dense layer(s)
		self.dense = tf.keras.layers.Dense(self.english_vocab_size, activation='softmax')

	@tf.function
	def call(self, encoder_input, decoder_input):
		"""
		:param encoder_input: batched ids corresponding to french sentences
		:param decoder_input: batched ids corresponding to english sentences
		:return prbs: The 3d probabilities as a tensor, [batch_size x window_size x english_vocab_size]
		"""
		# TODO:
		# 1) Add the positional embeddings to french sentence embeddings
		fre_embed = self.position_fre.call(tf.nn.embedding_lookup(self.EF, encoder_input))

		# 2) Pass the french sentence embeddings to the encoder
		encoder_out = self.encoder(fre_embed)

		# 3) Add positional embeddings to the english sentence embeddings
		eng_embed = self.position_eng.call(tf.nn.embedding_lookup(self.EE, decoder_input))

		# 4) Pass the english embeddings and output of your encoder, to the decoder
		decoder_out = self.decoder(eng_embed, encoder_out)

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
		# Note: you can reuse this from rnn_model.

		return tf.boolean_mask(tf.keras.losses.sparse_categorical_crossentropy(labels, prbs), mask)
