import numpy as np
import matplotlib.pyplot as plt

from numpy.core.defchararray import equal
from numpy.core.fromnumeric import repeat

#creating our Class
class NeuralNetwork:

    #importing inputs 
    def __init__ (self,num_inputs= 8,num_hidden= 3,num_outputs=8):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        self.error_history = []
        self.epoch_list = []
        #setting our layers
        layers = [num_inputs] + [num_hidden] + [num_outputs]

        #random weights and biases
        weights = []
        biases = []
        for i in range(len(layers)-1):
            w = np.random.rand(layers[i], layers[i+1])
            weights.append(w)
            b = np.zeros(layers[i])
            biases.append(b)
        self.weights = weights
        self.biases = biases
        #print("Random Weights: ",weights)
        #print("Random Biases: ",biases)
            

        #create activations 
        activations =[]
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations    

        #create derivatives
        derivatives =[]
        for i in range(len(layers)-1):
            d = np.zeros((layers[i], layers[i+1]))
            derivatives.append(d)
        self.derivatives = derivatives  

    #implement forwardpropagation
    def forward_propagate(self,inputs):
        
        # the input layer activation is just the input itself
        activations = inputs
        # save the activations for backpropogation
        self.activations[0] = inputs

        for i, w in enumerate(self.weights):
            #calculate the net inputs
            net_inputs = np.dot(activations, w)
            #calculate the activations
            activations = self.sigmoid(net_inputs)
            # save the activations for backpropogation
            self.activations[i + 1] = activations
        return activations

    #implement backpropagation
    def back_propagation(self, error, testing=False):
         # iterate backwards through the network layers
        for i in reversed(range(len(self.derivatives))):
            # get activation for previous layer
            activations =self.activations[i+1]
            # apply sigmoid derivative function for delta and reshape it
            delta = error *self.sigmoid_derivative(activations)
            delta_new = delta.reshape(delta.shape[0], -1).T
             # get activations for current layer and reshape
            current_activations = self.activations[i]
            current_activations_new = current_activations.reshape(current_activations.shape[0], -1)
            # save derivative after applying matrix multiplication
            self.derivatives[i] = np.dot(current_activations_new,delta_new)
            # backpropogate the next error
            error = np.dot(delta, self.weights[i].T)

            #if testing:
                #print("Derivatives for Weights{}: {}".format(i,self.derivatives[i]))
                #print("Activations for Layer{}: {}".format(i,self.activations[i]))
        return error 


    #implement gradient descent 
    def gradient_descent(self,learning_rate):
        for i in range(len(self.weights)):
            weights = self.weights[i]
            #print("First weights{} {}".format(i,weights))
            derivatives = self.derivatives[i]
            weights += derivatives*learning_rate
            #print("New weights{} {}".format(i,weights))

    #derivativation of sigmoid ==> f'(x) = f(x)(1-f(x))
    def sigmoid_derivative(self,x):
        return x*(1.0 -x)
    
    #activation of the sigmoid function ==> f(x) = 1/(1+e^(-x)) 
    def sigmoid(self,x):
        return 1.0/(1+np.exp(-x))

    # train the neural net
    def train(self,inputs,targets,repeats,learning_rate):
        for i in range(repeats):
            sum_error = 0
            for input,target_output in zip(inputs,targets):    
                #forward propagation
                outputs = self.forward_propagate(input)
                #calculate the error
                error = target_output - outputs
                #back propagation, for testing=True
                self.back_propagation(error)
                 #apply gradient descent 
                self.gradient_descent(learning_rate)
                sum_error += self._mse(target_output,outputs)
                # keep track of the error history over each epoch
            self.error_history.append(np.average(np.abs(error)))
            self.epoch_list.append(i)
                

            #report error for each repeat         
            #print("Error: {} at epoch {}".format(sum_error / len(items), i+1))


    #Mean Squared Error loss function
    def _mse(self,target_output,outputs):
        return np.average((target_output - outputs)**2)
        

if __name__ == "__main__":
    
    # create a dataset to train the network 
    items = np.array([[1,0,0,0,0,0,0,0],
         [0,1,0,0,0,0,0,0],
         [0,0,1,0,0,0,0,0],
         [0,0,0,1,0,0,0,0],
         [0,0,0,0,1,0,0,0],
         [0,0,0,0,0,1,0,0],
         [0,0,0,0,0,0,1,0],
         [0,0,0,0,0,0,0,1]])

    #setting our target output
    targets = items

    #creat our NN
    NN = NeuralNetwork(8, 3, 8)
    

    #train our NN with 5000 repeats and 0.04 learning rate
    NN.train(items,targets, 100000, 0.8)

    # create two new examples to predict                                   
    example = np.array([0, 0, 0, 0, 0, 0, 0, 1])
    example_output = example
    example_2 = np.array([0, 1, 0, 0, 0, 0, 0, 0]) 
    example_2_output = example_2

    #round the results to see them clearly
    output = np.around(NN.forward_propagate(example))
    output_2 = np.around(NN.forward_propagate(example_2))
    print()
    print()
    #predictions
    print("With 10000 repeats and 0.4 learning rate")
    print("We want to predict that {} is the same as {}".format(example , example_output))
    print("Our Network prediction: is that {} is the same as {}".format(example , output))
    print("We want to predict that {} is the same as {}".format(example_2 , example_2_output))
    print("Our Network prediction: is that {} is the same as {}".format(example_2 , output_2))
    

    #print results
    #print('Inputs: ', items)
    #print ('Target: ', targets)
    #print('Outputs: ', outputs)

    # plot the error over the entire training duration
    plt.figure(figsize=(10,5))
    plt.plot( NN.epoch_list, NN.error_history)
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.show()