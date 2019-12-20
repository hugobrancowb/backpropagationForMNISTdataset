# Neural Network using Backpropagation to classify images into 1 out of 10 categories
Backpropagation algorithm written in C to train and classify handwritten numbers (0-9) from [MNIST](http://yann.lecun.com/exdb/mnist/) dataset.

![Sample image from dataset.](sample.png)

## Dataset
Although the original dataset contains 70 thousand samples, this algorithm uses only 6 thousand images for training (```test-6k-images-labels```) and 4 thousand for testing (```test-4k-images-labels```) the network.

## Build
```gcc final.c -o final.x -Wall -Wextra -g -std=c99 -lm -O0 -fPIC```

## Run
```
-h  help
-t  train a neural network
-r  run the saved neural network
```