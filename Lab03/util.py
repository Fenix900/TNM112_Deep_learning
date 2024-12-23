import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras

# Model evaluation with Keras model
def evaluate(model, dataset, final=False):
    print('Model performance:')
    
    # Provides loss and accuracy on the training set
    score = model.evaluate(dataset.x_train, dataset.y_train_oh, verbose=False)
    print('\tTrain loss:          %0.4f'%score[0])
    print('\tTrain accuracy:      %0.2f'%(100*score[1]))

    # If there are more metrics, we assume AUC
    if len(score) > 2:
        print('\tTrain AUC:           %0.2f'%(score[2]))

    # Provides loss and accuracy on the test set
    if final:
        score = model.evaluate(dataset.x_test, dataset.y_test_oh, verbose=False)
        print('\tTest loss:           %0.4f'%score[0])
        print('\tTest accuracy:       %0.2f'%(100*score[1]))
        
        if len(score) > 2:
            print('\tTest AUC:            %0.2f'%(score[2]))

    # Provides loss and accuracy on the validation set
    else:
        score = model.evaluate(dataset.x_valid, dataset.y_valid_oh, verbose=False)
        print('\tValidation loss:     %0.4f'%score[0])
        print('\tValidation accuracy: %0.2f'%(100*score[1]))

        if len(score) > 2:
            print('\tValidation AUC:     %0.2f'%(score[2]))

    return score

# Plotting of training history
def plot_training(log):
    N_train = len(log.history['loss'])
    N_valid = len(log.history['val_loss'])
    
    plt.figure(figsize=(18,4))
    
    # Plot loss on training and validation set
    plt.subplot(1,2,1)
    plt.plot(log.history['loss'])
    plt.plot(np.linspace(0,N_train-1,N_valid), log.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.grid('on')
    plt.legend(['Train', 'Validation'])
    
    # Plot accuracy on training and validation set
    plt.subplot(1,2,2)
    plt.plot(log.history['accuracy'])
    plt.plot(np.linspace(0,N_train-1,N_valid), log.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.grid('on')
    plt.legend(['Train', 'Validation'])
    
    plt.show()
