################################################################################
# CSE 253: Programming Assignment 2
# Code snippet by Eric Yang Yu, Ajit Kumar, Savyasachi
# Winter 2022
################################################################################

import numpy as np
import math


class Activation:
    """
    The class implements different types of activation functions for
    your neural network layers.

    Example (for sigmoid):
        >>> sigmoid_layer = Activation("sigmoid")
        >>> z = sigmoid_layer(a)
        >>> gradient = sigmoid_layer.backward(delta)
    """

    def __init__(self, activation_type="sigmoid"):
        """
        Initialize activation type and placeholders here.
        """
        if activation_type not in ["sigmoid", "tanh", "ReLU"]:
            raise NotImplementedError("%s is not implemented." % (activation_type))

        # Type of non-linear activation.
        self.activation_type = activation_type
        # Placeholder for input. This will be used for computing gradients.
        # Remember from my write up that a is a J*N matrix, where J is number of units being connected 
        # and N is number of data points in our minibatch.
        self.x = None
        
        # State the type of the layer to differentiate from normal layer
        self.type="activation"

    def __call__(self, a):
        """
        This method allows your instances to be callable.
        """
        # We need to save this vector, a, for computing the gradient evaluated at a.
        self.x=a
        return self.forward(a)

    def forward(self, a):
        """
        Compute the forward pass.
        """
        
        
        if self.activation_type == "sigmoid":
            return self.sigmoid(a)

        elif self.activation_type == "tanh":
            return self.tanh(a)

        elif self.activation_type == "ReLU":
            return self.ReLU(a)

    def backward(self, delta):
        """
        Compute the backward pass.
        """
        if self.activation_type == "sigmoid":
            grad = self.grad_sigmoid()

        elif self.activation_type == "tanh":
            grad = self.grad_tanh()

        elif self.activation_type == "ReLU":
            grad = self.grad_ReLU()

        return grad * delta

    def sigmoid(self, x):
        """
        Implement the sigmoid activation here.
        Notice that I implement the clipping to avoid numerical overflow 
        """
        x=np.clip(x,-20,20)
        return 1/(1+np.exp(-x))

    def tanh(self, x):
        """
        Implement tanh here.
        """
        return 1.7159*np.tanh((2/3)*x)
        

    def ReLU(self, x):
        """
        Implement ReLU here.
        """
        # Apply element-wise relu to a given matrix element-wise
       
        return np.clip(self.x,0,a_max=200)

    def grad_sigmoid(self):
        """
        Compute the gradient for sigmoid here.
        """
        return self.sigmoid(self.x)*(1-self.sigmoid(self.x))

    def grad_tanh(self):
        """
        Compute the gradient for tanh here.
        """
        return 1-np.square(np.tanh(self.x))

    def grad_ReLU(self):
        """
        Compute the gradient for ReLU here.
        """
        original_shape=self.x.shape
        x=self.x.flatten()
        out=[]
        for i in range(len(x)):
            if x[i]>=0:
                out.append(1)
            else:
                out.append(0)
        out=np.array(out)
        return out.reshape(original_shape)


class Layer:
    """
    This class implements Fully Connected layers for your neural network.

    Example:
        >>> fully_connected_layer = Layer(1024, 100)
        >>> output = fully_connected_layer(input)
        >>> gradient = fully_connected_layer.backward(delta)
    """

    def __init__(self, in_units, out_units):
        """
        Define the architecture and create placeholder.
        """        
        
        np.random.seed(42)
        self.w = math.sqrt(2 / in_units) * np.random.randn(in_units,
                                                           out_units)  # You can experiment with initialization.
        # Notice that the dimension 1*M is suffice due to the fact array can be broadcasted
        self.b = np.random.randn(1, out_units).flatten()  # Create a placeholder for Bias; 
        self.x = None  # Save the input to forward in this; This is dimension of N*j.
        self.a = None  # Save the output of forward pass in this (without activation)

        self.d_x = None  # Save the gradient w.r.t x in this
        self.d_w = None  # Save the gradient w.r.t w in this
        self.d_b = None  # Save the gradient w.r.t b in this


        self.prev_d_w=None
        
        # State the type of the layer to differentiate from activation layer
        self.type="layer"

    def __call__(self, x):
        """
        Make layer callable.
        """
        self.x=x
        return self.forward(x)

    def forward(self, x):
        """
        Compute the forward pass through the layer here.
        Do not apply activation here.
        Return self.a
        """
        
        self.a=np.matmul(self.x,self.w)+self.b # save the value for a;
        
        
        return self.a
    def backward(self, delta):
        """
        Write the code for backward pass. This takes in gradient from its next layer as input,
        computes gradient for its weights and the delta to pass to its previous layers.
        Return self.dx
        """
        
        # the delta matrix from next layer have the dimentionality of N*F
        # Right after ariive this layer they become N*M 
        self.prev_d_w=self.d_w # we define this for the purpose of apply momentum 
        self.d_w=np.matmul(self.x.T,delta) 
        
        

        # Now let's compute the d_b=delta*1
        # Its' vectorized form is 1*n vector(all ones since it's bias) *  n*M delta.
        bias_units=np.ones((1, len(self.x))).flatten()
        self.d_b=np.matmul(bias_units,delta)
        

        # Then we should compute the weighted sum of the delta and pass it to the activation layer 
        # In previous layer(activation layer). The full delta will be computed.
        # delta:N*M w:J*M (The notation switch here )
        # store weighted sum of delta as dx.
        
        
        self.d_x=np.matmul(delta,self.w.T)

        return self.d_x


class NeuralNetwork:
    """
    Create a Neural Network specified by the input configuration.

    Example:
        >>> net = NeuralNetwork(config)
        >>> output = net(input)
        >>> net.backward()
    """

    def __init__(self, config):
        """
        Create the Neural Network using config.
        """
        self.layers = []  # Store all layers in this list.
        self.x = None  # Save the input to forward in this
        self.y = None  # Save the output vector of model in this
        self.targets = None  # Save the targets in forward in this variable

        # Add layers specified by layer_specs.
        for i in range(len(config['layer_specs']) - 1):
            self.layers.append(Layer(config['layer_specs'][i], config['layer_specs'][i + 1]))
            if i < len(config['layer_specs']) - 2:
                self.layers.append(Activation(config['activation']))

    def __call__(self, x, targets=None):
        """
        Make NeuralNetwork callable.
        """
        
        return self.forward(x, targets)

    def forward(self, x, targets=None):
        """
        Compute forward pass through all the layers in the network and return it.
        If targets are provided, return loss as well.
        """
        
        # Save the data here. 
        self.x=x
        self.targets=targets

        # Now we are going to loop through each and every layer of our network. 
        # our input signal will pass through these successive networks 
        for i in self.layers:
            out=i(self.x) # Compute the output signal from a layer
            self.x=out # Store the output signal for latar use 
            
            

        # Remeber that the last layer of the network is the output layer, which must
        # be gone through softmax transformation. 
        self.y=self.softmax(self.x)
        
        
        if targets is not None:
            loss=self.loss(self.y,self.targets)
            return (self.y,loss)

        return self.y

    def backward(self):
        """
        Implement backpropagation here.
        Call backward methods of individual layer's.
        """

        # Get the necessary parameters for optimization. 
        learning_rate=config["learning_rate"]
        L2_penalty=config["L2_penalty"]
        if_momentum=config["momentum"]
        momentum_gamma=config["momentum_gamma"]

        # before we start, we can compute the error signal. 
        error_signal=self.targets-self.y
        
        for i in range(len(self.layers)):
            
            # Use negative indexing to travel backward.
            current_layer=self.layers[-(i+1)]
            # Feed in the current error signal and update error signal(delta) to the previous layer 
            # Error signal here is just the weighted sum of delta
            error_signal=current_layer.backward(error_signal)
            # Update the weights and bias term based on gradient descent 
            # During the update, per specification of the user, momentum and L2 regularization will be involved
            # So, our new weight change: Gamma*old_weight_change+(1-Gamma)*new_weight_change-2*lambda*w
            
            # Need to notice that we only update the weights if the layer is not activation layer.
            if current_layer.type=="layer":
                if if_momentum:
                    if current_layer.prev_d_w is None: # This happens during the first weight update 
                        # Don't be confused by minus sign here. I do this becasue w=w+a*d_w; all signs are just flipped
                        combined_weight_change=current_layer.d_w-2*L2_penalty*current_layer.w
                    else:  
                        combined_weight_change=momentum_gamma*current_layer.prev_d_w+(1-momentum_gamma)*current_layer.d_w-2*L2_penalty*current_layer.w
                else:
                    # Ignore the momentum term all the time; 
                    combined_weight_change=current_layer.d_w-2*L2_penalty*current_layer.w

                current_layer.w=current_layer.w+learning_rate*combined_weight_change


    def softmax(self, x):
        """
        Implement the softmax function here.
        Remember to take care of the overflow condition.
        """
        a=x
        a=np.clip(a,-20,20)
        a_exp = np.exp(a)
        partition = np.sum(a_exp, axis=1).reshape(-1,1)
        return a_exp / partition

    def loss(self, logits, targets):
        """
        Compute the categorical cross-entropy loss and return it.
        """
        # We reuse the code for logistic regression.
        y=logits
        t=targets
        entropy=np.zeros(t.shape[0])
        for i in range(10):
            target_column=t[:,i]
            prediction_column=y[:,i]
            entropy+=target_column*np.log(prediction_column)
        return -np.mean(entropy/10)