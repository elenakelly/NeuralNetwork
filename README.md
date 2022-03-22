## A simple Implementation of a Neural Network and of Backpropagation.
The goal is to produce and train a network with 3 layers:  input - hidden - output. <br />
Both the input layer and the output layer will have 8 nodes and the hidden layer only 3 nodes.<br />
The learning examples will each have 7 zeros and 1 one in them (so there will be only 8 different learning examples). <br />
In the output the network should learn exactly the same as the input.<br />
So when the input layer is : <0,0,0,1,0,0,0,0> , the output to aim for is also: <0,0,0,1,0,0,0,0>. <br />
The network is trying to learn on the 8 different learning examples with a reproducing function.<br />
Then study the weights and the activations of the hidden nodes of network and try to interpret them. 
<img width="685" alt="Screenshot 2022-03-22 at 5 04 35 PM" src="https://user-images.githubusercontent.com/59971317/159525964-e5960a71-6393-4f82-a0eb-43894056e512.png">


### Initialization
First we initialise the matrixes for the three layers with random weights between 0 and 1 and all biases as 0. <br />
Then we implement the forward propagation method that sends the sum of the weights, times the input values through the sigmoid function.<br />
The backpropagation method iterates backwards through the network layers and calculates the delta cumulatively.<br />
Then saves it for each weight when finally the gradient descent method updates the weights.<br />

### Train
To train the network, we first initialise the 8 , 3 and 8 neurons for each of the 3 layers respectively.<br /> 
For a batch of 8 samples carrying all the possible different value combination, for each of input we forward propagate to the hidden and output layers, then calculate the error as the difference between the target output and the sample output.<br /> 
We backpropagate it through the layers, and apply the gradient descent method with a parameter tuning the learning rate.<br /> 
This process happens for a number of epochs specified in the initalisation. <br />

### Results
Finally we keep track of the cumulative error rate per epoch for tracking and plotting reasons.<br />
The number of epochs as well as the learning rates are hard coded in the main method of the code and two hard coded examples of predictions are exemplified after the training has finished.


## Installation
The program is in Python <br />
In order to use the code you need to install only Numpy and Matplot packages
   ```sh
    import numpy as np
    import matplotlib.pyplot as plt
   ```


