# Convolutional Neural Networks on CIFAR2


Part 1:
My final network contains 4 CNN layers and 4 dense layers.
Each CNN layer contains conv2d, batch normalization, relu and max pooling.
Each dense layer contains relu and dropout(except the last dense layer).
For every epoch, train set are processed by batch with batch size= 200.

Part 2:
If the filter size is odd, my conv2d function can work well.

When I using GPU in my own laptop, it takes about 1 minute to train and test.
When I using CPU, it takes about 8 minutes to train the model and 5 minutes to test using my own conv2d function.

After 10 epochs of training, my final accuracy on test set is about 0.77 - 0.8.
