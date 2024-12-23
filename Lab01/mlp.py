import numpy as np
import data_generator

# Different activations functions
def activation(x, activation):
    
    #TODO: specify the different activation functions (done :))
    # 'activation' could be: 'linear', 'relu', 'sigmoid', or 'softmax'
    if activation == 'linear':
        return x
    elif activation == "sigmoid":
        return 1/(1+np.exp(-x))
    elif activation == "relu":
        return np.maximum(0, x)
    elif activation == "softmax":
        return np.exp(x)/np.sum(np.exp(x),axis=1, keepdims=True)
    else:
        raise Exception("Activation function is not valid", activation)

#-------------------------------
# Our own implementation of an MLP
#-------------------------------
class MLP:
    def __init__(
        self,
        dataset,         # DataGenerator
    ):
        self.dataset = dataset

    # Set up the MLP from provided weights and biases
    def setup_model(
        self,
        W,                   # List of weight matrices
        b,                   # List of bias vectors
        activation='linear'  # Activation function of layers
    ):
        self.activation = activation

        # TODO: specify the number of hidden layers based on the length of the provided lists
        self.hidden_layers = len(W)-1

        self.W = W
        self.b = b

        # TODO: specify the total number of weights in the model (both weight matrices and bias vectors)
        self.N = 0
        for i in range(len(self.W)):
            self.N += np.prod(self.W[i].shape) + np.prod(self.b[i].shape)

        print('Number of hidden layers: ', self.hidden_layers)
        print('Number of model weights: ', self.N)

    # Feed-forward through the MLP
    def feedforward(
        self,
        x      # Input data points
    ):
        # TODO: specify a matrix for storing output values
       

        # TODO: implement the feed-forward layer operations
        # 1. Specify a loop over all the datapoints
        # 2. Specify the input layer (2x1 matrix)
        # 3. For each hidden layer, perform the MLP operations
        #    - multiply weight matrix and output from previous layer
        #    - add bias vector
        #    - apply activation function
        # 4. Specify the final layer, with 'softmax' activation
        y = x
    
        for index in range(self.hidden_layers):
            weightTransposed = self.W[index].T
            z = np.matmul(y, weightTransposed) + np.squeeze(self.b[index])
            y = activation(z, self.activation)
        #Apply last layers multiplication to get two classes to give to softmax
        z_final = np.matmul(y, self.W[-1].T) + np.squeeze(self.b[-1])
        y = activation(z_final, "softmax")
        return y

    # Measure performance of model
    def evaluate(self):
        print('Model performance:')

        predictions_of_x_train = self.feedforward(self.dataset.x_train)
        print("predictions_of_x_train _ ", predictions_of_x_train.shape)
        predictions_of_x_test = self.feedforward(self.dataset.x_test)
        # TODO: formulate the training loss and accuracy of the MLP
        # Assume the mean squared error loss
        # Hint: For calculating accuracy, use np.argmax to get predicted class

        train_loss = np.mean((predictions_of_x_train - self.dataset.y_train_oh)**2)
        train_correct = sum(np.argmax(predictions_of_x_train,axis=1) == self.dataset.y_train)
        train_acc = train_correct/len(self.dataset.x_train)
        print("\tTrain loss:     %0.4f"%train_loss)
        print("\tTrain accuracy: %0.4f"%train_acc)

        # TODO: formulate the test loss and accuracy of the MLP
        test_loss = np.mean((predictions_of_x_test - self.dataset.y_test_oh)**2)
        test_correct = sum(np.argmax(predictions_of_x_test,axis=1) == self.dataset.y_test)
        test_acc = test_correct/len(self.dataset.x_test)
        print("\tTest loss:      %0.4f"%test_loss)
        print("\tTest accuracy:  %0.4f"%test_acc)