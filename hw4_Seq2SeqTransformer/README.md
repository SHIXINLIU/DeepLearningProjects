# Machine Translation
RNN and TRANSFORMER method to translate French to English. 

## Preprocess
Run 'pip install gast'.
* Read the French training data and the English training data
* Read the French testing data and the English testing data
* Call pad_corpus on the training and testing data
* Build the French and English vocabularies from the training data, then use the vocabularies to convert the sentences to ID form
* Return the processed data, dictionaries and the English padding ID

## part1.RNN
run `python assignment.py RNN`
My final perplexity = 8, accuracy = 0.66

## part2.TRANSFORMER
run `python assignment.py TRANSFORMER`
My final perplexity = 3, accuracy = 0.80

