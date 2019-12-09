# Language Model: Trigram and RNN
NLP and predict expecting word in a sentence. 

Since the way we deal with dataset is nonsense, the learning is nonsense and final result will be ridiculious.

## Preprocess
* Load the train words and split the words on whitespace.
* Load the test words and split the words on whitespace.
* Create a vocab dict that maps a unique index (id) to each word in the corpus
* Convert the list of training and test words to their indeces, making a 1-d list/array for each
* Return an iterable of training ids, an iterable of testing ids, and the vocab dict
see `preprocess.py`

## Trigram
Run `python trigram.py` to train and generate a silly sentence after training.
embedding_size = 256, learning_rate = 5e-4
My final perplexity is around 250

## RNN (LSTM)
This time I can use tf.keras.layers! 
Run `python rnn.py`.
embedding_size = 300, rnn_size = 256, learning_rate = 0.01
My final perplexity is around 140

## Conclusion
Both models can be trained within 10 minutes with no bugs. 
The generated sentence will be always like "I like UNK- of UNK- and UNK- of UNK- UNK-".
