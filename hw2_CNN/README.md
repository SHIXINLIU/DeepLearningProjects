# Convolutional Neural Networks on CIFAR2
This time we use CIFAR dataset to train CNN to judge cat or dog...

## Dataset
While the CIFAR10 dataset has 10 possible classes (airplane, automobile, bird, cat, deer, frog, horse, ship, and truck), you will build a CNN to take in an image and correctly predict its class to either be a cat or dog, hence CIFAR2. So after reading in the data, I only pick 2 classes to train and test. 

## Part1: use tf.nn.conv2d (not allowed to use keras yet!)
Built CNN using tf.nn.conv2d tiredly. see `/assignment.py`

My network contains 4 CNN layers and 4 dense layers. 

Each CNN layer contains conv2d, batch normalization, relu and max pooling. 

Each dense layer contains relu and dropout(except the last dense layer).
 

## Part 2: implement conv2d manually
Use math knowledge and tf.tensordot func. see `convolution.py`

If the filter size is odd, my conv2d function can work well. So lazy to adapt to the cases of even filter size. 

When I using GPU in my own laptop, it takes about 1 minute to train and test.

When I using CPU, it takes about 8 minutes to train the model and 5 minutes to test.

## Conclusion
After 10 epochs of training, my final accuracy on test set is about 0.77 - 0.8.
