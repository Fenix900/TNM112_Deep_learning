import numpy as np
from scipy import signal
import skimage
import data_generator

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor

# Different activations functions
def activation(x, activation):
    
    #TODO: specify the different activation functions
    # 'activation' could be: 'linear', 'relu', 'sigmoid', or 'softmax'
    if activation == 'linear':
        return x
    elif activation == "sigmoid":
        return 1/(1+np.exp(-x))
    elif activation == "relu":
        return np.maximum(0, x)
    elif activation == "softmax":
        return np.exp(x)/np.sum(np.exp(x), keepdims=True)
    else:
        raise Exception("Activation function is not valid", activation)

# 2D convolutional layer
def conv2d_layer(h,     # activations from previous layer, shape = [height, width, channels prev. layer]
                 W,     # conv. kernels, shape = [kernel height, kernel width, channels prev. layer, channels this layer]
                 b,     # bias vector
                 act    # activation function
):
    # TODO: implement the convolutional layer
    # 1. Specify the number of input and output channels
    CI = W.shape[2] # Number of input channels
    CO = W.shape[3] # Number of output channels
    #if h.shape[-1] == 1: #if h has a 1 dimension in the end----
        #h = np.squeeze(h) # h as input is (28,28,1) but we want to have it (28,28) so we can conv. it---
    
    # 2. Setup a nested loop over the number of output channels 
    #    and the number of input channels
    x_out,y_out = h[:,:,0].shape
    output = np.zeros((x_out,y_out,CO))
    
    for i in range(CO):
        conv_sum = np.zeros(h[:,:,0].shape)
        for j in range(CI):
            
    # 3. Get the kernel mapping between channels i and j
            kernel = W[:, :, j, i]
    # 4. Flip the kernel horizontally and vertically (since
    #    We want to perform cross-correlation, not convolution.
    #    You can, e.g., look at np.flipud and np.fliplr
            flippedKernel = np.flipud(np.fliplr(kernel))
    # 5. Run convolution (you can, e.g., look at the convolve2d
    #    function in the scipy.signal library)
            conv_result = signal.convolve2d(h[:, :, j], flippedKernel, mode="same")
    # 6. Sum convolutions over input channels, as described in the 
    #    equation for the convolutional layer
            conv_sum += conv_result
    # 7. Finally, add the bias and apply activation function
        conv_sum +=  b[i]
        output[:, :, i] = activation(conv_sum, act)

    return output


# 2D max pooling layer
def pool2d_layer(h):  # activations from conv layer, shape = [height, width, channels]
    # TODO: implement the pooling operation
    # 1. Specify the height and width of the output
    sx, sy, c = h.shape
    sy = int(sy/2)
    sx = int(sx/2)

    # 2. Specify array to store output
    h_out = np.zeros((sx, sy, c))

    # 3. Perform pooling for each channel.
    #    You can, e.g., look at the measure.block_reduce() function
    #    in the skimage library
    for i in range(c):
        for j in range(sx):
            for k in range(sy):
                #print("size is ", 2*j, " to ", 2*j+1, " by ", 2*k ," to ", 2*k+1, " and I is ", i)
                pooling_values = h[2*j:2*j+2, 2*k:2*k+2, i] #2x2 matrix to choose the biggest
                #print("pooling_values: ", pooling_values)
                max_value = np.max(pooling_values)
                h_out[j,k,i] = max_value
    return h_out


# Flattening layer
def flatten_layer(h): # activations from conv/pool layer, shape = [height, width, channels]
    # TODO: Flatten the array to a vector output.
    # You can, e.g., look at the np.ndarray.flatten() function
    return h.flatten()
    
# Dense (fully-connected) layer
def dense_layer(h,   # Activations from previous layer
                W,   # Weight matrix
                b,   # Bias vector
                act  # Activation function
):
    # TODO: implement the dense layer.
    # You can use the code from your implementation
    # in Lab 1. Make sure that the h vector is a [Kx1] array.
    z = np.matmul(W,h) + np.squeeze(b)
    y = activation(z, act)
    return y
    
#---------------------------------
# Our own implementation of a CNN
#---------------------------------
class CNN:
    def __init__(
        self,
        dataset,         # DataGenerator
        verbose=True     # For printing info messages
    ):
        self.verbose = verbose
        self.dataset = dataset

    # Set up the CNN from provided weights
    def setup_model(
        self,
        W,                   # List of weight matrices
        b,                   # List of bias vectors
        lname,               # List of layer names
        activation='relu'    # Activation function of layers
    ):
        self.activation = activation
        self.lname = lname

        self.W = W
        self.b = b

        # TODO: specify the total number of weights in the model
        #       (convolutional kernels, weight matrices, and bias vectors)
        self.N = 0
        for i, layer_name in enumerate(self.lname):
            if layer_name == 'conv' or layer_name == 'dense':
                weight_shape = W[i].shape  # weights for the layer (l)
                num_weights = np.prod(weight_shape)  # Get total number by multiplying the dimensions
                num_biases = len(b[i])           # Number of biases
                self.N += num_weights + num_biases
        print('Number of model weights: ', self.N)

    # Feedforward through the CNN of one single image
    def feedforward_sample(self, h):

        # Loop over all the model layers
        for l in range(len(self.lname)):
            act = self.activation
            if self.lname[l] == 'conv':
                h = conv2d_layer(h, self.W[l], self.b[l], act)
            elif self.lname[l] == 'pool':
                h = pool2d_layer(h)
            elif self.lname[l] == 'flatten':
                h = flatten_layer(h)
            elif self.lname[l] == 'dense':
                if l==(len(self.lname)-1):
                    act = 'softmax'
                h = dense_layer(h, self.W[l], self.b[l], act)
        return h

    # Feedforward through the CNN of a dataset
    def feedforward(self, x):
        # Output array
        y = np.zeros((x.shape[0],self.dataset.K))

        # Go through each image
        for k in range(x.shape[0]):
            if self.verbose and np.mod(k,1000)==0:
                print('sample %d of %d'%(k,x.shape[0]))

            # Apply layers to image
            y[k,:] = self.feedforward_sample(x[k])   
            
        return y

    # Measure performance of model
    def evaluate(self):
        print('Model performance:')

        # TODO: formulate the training loss and accuracy of the CNN.
        # Assume the cross-entropy loss.
        # For the accuracy, you can use the implementation from Lab 1.
        
        print("first: ", self.dataset.x_train.shape)
        pred_x_train = self.feedforward(self.dataset.x_train) #Calculate the predicted x train samples
        pred_x_test = self.feedforward(self.dataset.x_test) #Calculate the predicted x test samples
        
        #Remove all very small predictions to aviod log of 0 or somthing like that
        epsilon = 1e-10
        pred_x_train = np.clip(pred_x_train, epsilon, 1. - epsilon)
        pred_x_test = np.clip(pred_x_test, epsilon, 1. - epsilon)

        #Train loss
        train_loss = (-np.sum(self.dataset.y_train_oh * np.log(pred_x_train)))/self.dataset.N_train
        #Train accuracy
        train_correct = sum(np.argmax(pred_x_train, axis=1) == self.dataset.y_train)
        train_acc = train_correct/self.dataset.N_train
        print("\tTrain loss:     %0.4f"%train_loss)
        print("\tTrain accuracy: %0.4f"%train_acc)

        # TODO: formulate the test loss and accuracy of the CNN
        #Test loss
        test_loss = (-np.sum(self.dataset.y_test_oh * np.log(pred_x_test)))/len(self.dataset.x_test)
        #Test accuracy
        test_correct = sum(np.argmax(pred_x_test, axis=1) == self.dataset.y_test)
        test_acc = test_correct/len(self.dataset.x_test)
        print("\tTest loss:      %0.4f"%test_loss)
        print("\tTest accuracy:  %0.4f"%test_acc)
