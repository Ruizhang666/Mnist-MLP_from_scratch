################################################################################
# CSE 151b: Programming Assignment 2
# Code snippet by Eric Yang Yu, Ajit Kumar, Savyasachi
# Winter 2022
################################################################################
from data import write_to_file
from neuralnet import *
import matplotlib.pyplot as plt
import time
from copy import deepcopy

#generate minibatches
def generate_minibatches(dataset, batch_size=64):
    X, y = dataset
    l_idx, r_idx = 0, batch_size
    while r_idx < len(X):
        yield X[l_idx:r_idx], y[l_idx:r_idx]
        l_idx, r_idx = r_idx, r_idx + batch_size
    yield X[l_idx:], y[l_idx:]

#shuffle the dataset
def shuffle(dataset):
    """
    Shuffle dataset.
    Make sure that corresponding images and labels are kept together. 
    Ideas: 
        NumPy array indexing 
            https://numpy.org/doc/stable/user/basics.indexing.html#advanced-indexing
    Parameters
    ----------
    dataset
        Tuple containing
            Images (X)
            Labels (y)
    Returns
    -------
        Tuple containing
            Images (X)
            Labels (y)
    """
    dim=dataset[0].shape[0]
    index1=np.arange(dim)
    np.random.shuffle(index1)
    shuffled_data=dataset[0][index1]
    shuffled_label=dataset[1][index1]
    return (shuffled_data,shuffled_label)

#one hot encoding
def one_hot_encoding(labels, num_classes=10):
    """
    Encode labels using one hot encoding and return them.
    """
    values = labels
    n_values = np.max(labels) + 1
    return np.eye(n_values)[labels] 

def train(x_train, y_train, x_val, y_val, config, experiment=None):
    """
    Train your model here using batch stochastic gradient descent and early stopping. Use config to set parameters
    for training like learning rate, momentum, etc.

    Args:
        x_train: The train patterns
        y_train: The train labels; 
        x_val: The validation set patterns
        y_val: The validation set labels
        config: The configs as specified in config.yaml
        experiment: An optional dict parameter for you to specify which experiment you want to run in train.

    Returns:
        5 things:
            training and validation loss and accuracies - 1D arrays of loss and accuracy values per epoch.
            best model - an instance of class NeuralNetwork. You can use copy.deepcopy(model) to save the best model.
    """
    train_acc = []
    val_acc = []
    train_loss = []
    val_loss = []
    best_model = None

    model = NeuralNetwork(config=config)

    # Read in some of the model parameters:
    epochs=config["epochs"]
    if_early_stop=config["early_stop"]
    early_stop_epoch=config['early_stop_epoch']
    batch_size=config["batch_size"]
    
    y_train=one_hot_encoding(y_train)
    train_dataset=(x_train,y_train)

    # Determine the number of iterations that will be used fortraining
    terations_num=None

    if if_early_stop:
        iterations_num=early_stop_epoch
    else:
        iterations_num=epochs
        

    for i in range(iterations_num):
        # shuffle the dataset and generate minibatchs. 
        train_dataset=shuffle(train_dataset)
        for X_train,y_train in generate_minibatches(train_dataset,batch_size=batch_size):
            # A forward pass to make prediction
            model(X_train,targets=y_train)
            # A backward pass to adjust the weights 
            model.backward()
        # Evaluate model's performance on training set. 
        predictions,loss=model.forward(x_train,targets=y_train)
        predicted_labels=np.argmax(predictions,axis=1)
        restored_labels=np.argmax(y_train,axis=1)
        single_accuracy=np.mean(predicted_labels== restored_labels)
        train_acc.append(single_accuracy)
        train_loss.append(loss)
        # Evaluate model's performance on validation set.
        val_oh=one_hot_encoding(y_val)
        p1,l2=model.forward(x_val,targets=val_oh)
        predicted=np.argmax(p1,axis=1)
        acc=np.mean(predicted==y_val)
        val_acc.append(acc)
        val_loss.append(l2)
    
    return train_acc, val_acc, train_loss, val_loss, model


def test(model, x_test, y_test):
    """
    Does a forward pass on the model and returns loss and accuracy on the test set.

    Args:
        model: The trained model to run a forward pass on.
        x_test: The test patterns.
        y_test: The test labels.

    Returns:
        Loss, Test accuracy
    """
    # return loss, accuracy
    y_test_oh=one_hot_encoding(y_test)
    test_prediction,test_loss=model.forward(x_test,targets=y_test_oh)
    test_labels_predicted=np.argmax(test_prediction,axis=1)
    return test_loss,np.mean(test_labels_predicted==y_test)


def train_mlp(x_train, y_train, x_val, y_val, x_test, y_test, config):
    """
    This function trains a single multi-layer perceptron and plots its performances.

    NOTE: For this function and any of the experiments, feel free to come up with your own ways of saving data
            (i.e. plots, performances, etc.). A recommendation is to save this function's data and each experiment's
            data into separate folders, but this part is up to you.
    """
    # train the model
    train_acc, valid_acc, train_loss, valid_loss, best_model = \
        train(x_train, y_train, x_val, y_val, config)

    test_loss, test_acc = test(best_model, x_test, y_test)

    print("Config: %r" % config)
    print("Test Loss", test_loss)
    print("Test Accuracy", test_acc)

    # DO NOT modify the code below.
    data = {'train_loss': train_loss, 'val_loss': valid_loss, 'train_acc': train_acc, 'val_acc': valid_acc,
            'best_model': best_model, 'test_loss': test_loss, 'test_acc': test_acc}

    write_to_file('./results.pkl', data)


def activation_experiment(x_train, y_train, x_val, y_val, x_test, y_test, config):
    """
    This function tests all the different activation functions available and then plots their performances.
    """
    #use sigmoid as the activation function
    config['activation'] = "sigmoid"
    train_acc, valid_acc, train_loss, valid_loss, best_model = train(x_train,y_train,x_val,y_val,config,experiment=None)
    test_loss, test_acc = test(best_model, x_test, y_test)
    return train_acc, valid_acc, train_loss, valid_loss,test_loss, test_acc

    #use relu as the activation function

    # config['activation'] = "ReLU"
    # train_acc, valid_acc, train_loss, valid_loss, best_model = train(x_train,y_train,x_val,y_val,config,experiment=None)
    # test_loss, test_acc = test(best_model, x_test, y_test)
    # return train_acc, valid_acc, train_loss, valid_loss,test_loss, test_acc


def topology_experiment(x_train, y_train, x_val, y_val, x_test, y_test, config):
    """
    This function tests performance of various network topologies, i.e. making
    the graph narrower and wider by halving and doubling the number of hidden units.

    Then, we change number of hidden layers to 2 of equal size instead of 1, and keep
    number of parameters roughly equal to the number of parameters of the best performing
    model previously.
    """
    # double the hidden units
    config['layer_specs'] = [784, 256, 10]
    train_acc, valid_acc, train_loss, valid_loss, best_model = train(x_train,y_train,x_val,y_val,config,experiment=None)
    test_loss, test_acc = test(best_model, x_test, y_test)
    return train_acc, valid_acc, train_loss, valid_loss,test_loss, test_acc

    # halve the hidden units

    # config['layer_specs'] = [784, 64, 10]
    # train_acc, valid_acc, train_loss, valid_loss, best_model = train(x_train,y_train,x_val,y_val,config,experiment=None)
    # test_loss, test_acc = test(best_model, x_test, y_test)
    # return train_acc, valid_acc, train_loss, valid_loss,test_loss, test_acc


def regularization_experiment(x_train, y_train, x_val, y_val, x_test, y_test, config):
    """
    This function tests the neural network with regularization.
    """
    train_acc, valid_acc, train_loss, valid_loss, best_model = train(x_train,y_train,x_val,y_val,config,experiment="regularization")
    test_loss, test_acc = test(best_model, x_test, y_test)
    return train_acc, valid_acc, train_loss, valid_loss,test_loss, test_acc



def check_gradients(x_train, y_train, config):
    """
    Check the network gradients computed by back propagation by comparing with the gradients computed using numerical
    approximation.
    """
    model = NeuralNetwork(config=config)
    #for one pattern
    x, y = x_train[5:6], y_train[5:6]
    error = 10e-2
    # input to hidden units is 0, hidden units to output is 2
    cur_layer = model.layers[0]
    
    # Initialize weight value to use
    weight = cur_layer.w[0,1]
    # weight = cur_layer.b[2]

    # add error
    cur_layer.w[0, 1] = weight + error
    # cur_layer.b[2] = weight + error
    predicted_y, loss_plus = model(x, y)

    # subtract error
    cur_layer.w[0,1] = weight - error
    # cur_layer.b[2] = weight - error
    predicted_y, loss_minus = model(x, y)
    
    # compute the approximated gradient and the actual
    approximated = (np.abs((loss_plus - loss_minus) / (2 * error)))
    model.backward()
    actual = np.abs(cur_layer.d_w[0,1])
    # actual = np.abs(cur_layer.d_b[2])/10
    
    #print out
    print("Approximated gradient: ",str(approximated), " Actual gradient: ", str(actual), " Difference: ",str(np.abs(actual - approximated)))