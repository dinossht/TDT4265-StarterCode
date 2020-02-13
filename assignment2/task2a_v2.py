import numpy as np
import utils
import typing
np.random.seed(1)

def sigmoid(X: np.ndarray, w: np.ndarray):
    z = np.exp(-X@w)
    return 1.0/(1.0+z)

def sigmoid_derivative(X: np.ndarray, w: np.ndarray):
    return sigmoid(X,w)*(1.0 - sigmoid(X,w))


def improved_sigmoid(X: np.ndarray, w: np.ndarray):
    #using the definition f(x) = 1.7159tanh(2x/3)
    z = (2/3)*X@w
    return 1.7159*np.tanh(z)

def improved_sigmoid_derivative(X: np.ndarray, w: np.ndarray):
    z = (2/3)*X@w
    return 1.7159*(2/3)*(1-np.tanh(z)**2)


def pre_process_images(X: np.ndarray, mean=128, std=50):
    """
    Args:
        X: images of shape [batch size, 784] in the range (0, 255)
    Returns:
        X: images of shape [batch size, 785]
    """
    assert X.shape[1] == 784,\
        f"X.shape[1]: {X.shape[1]}, should be 784"
    # Input normalization
    # Normalize using mean and standard deviation 
    mu      = mean
    sigma   = std 
    X_norm  = (X - mu) / sigma
    # Append 1 at the end (bias trick)
    return np.append(X_norm, np.ones((X.shape[0], 1)), axis=1)


def cross_entropy_loss(targets: np.ndarray, outputs: np.ndarray):
    """
    Args:
        targets: labels/targets of each image of shape: [batch size, num_classes]
        outputs: outputs of model of shape: [batch size, num_classes]
    Returns:
        Cross entropy error (float)
    """
    assert targets.shape == outputs.shape,\
        f"Targets shape: {targets.shape}, outputs: {outputs.shape}"
    ce = targets * np.log(outputs)
    N = outputs.shape[0]
    
    return (-1.0/(N))*np.sum(ce)
    raise NotImplementedError


class SoftmaxModel:

    def __init__(self,
                 # Number of neurons per layer
                 neurons_per_layer: typing.List[int],
                 use_improved_sigmoid: bool,  # Task 3a hyperparameter
                 use_improved_weight_init: bool,  # Task 3c hyperparameter
                 a_j: typing.List[float]
                 ):
        # Define number of input nodes
        self.I = 785
        self.use_improved_sigmoid = use_improved_sigmoid

        # Define number of output nodes
        # neurons_per_layer = [64, 10] indicates that we will have two layers:
        # A hidden layer with 64 neurons and a output layer with 10 neurons.
        self.neurons_per_layer = neurons_per_layer
        self.a_j = []

        # Initialize the weights
        self.ws = []
        prev = self.I
        for size in self.neurons_per_layer:
            w_shape = (prev, size)
            print("Initializing weight to shape:", w_shape)
            w = np.zeros(w_shape)
            #added:initilizing the weights
            if use_improved_weight_init == True:
                w = np.random.normal(0,1/np.sqrt(prev),(prev, size))
            else:
                w = np.random.uniform(-1, 1, (prev, size))
            self.ws.append(w)
            prev = size
        self.grads = [None for i in range(len(self.ws))]

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: images of shape [batch size, 785]
        Returns:
            y: output of model with shape [batch size, num_outputs]
        """
        #Hidden layer (using sigmoid):
        self.a_j = []
        self.a_j.append(X)
        for layer in range(0,(len(self.neurons_per_layer)-1)):
            if self.use_improved_sigmoid == True:
                self.a_j.append(improved_sigmoid(self.a_j[layer],self.ws[layer]))
            else:
                self.a_j.append(sigmoid(self.a_j[layer],self.ws[layer]))
        
        #Outer layer (using softmax):
        w_kj = self.ws[-1]
        z_k = np.exp(self.a_j[-1]@w_kj)
        sum_e = np.sum(z_k,1);
        output = (z_k.T/sum_e).T
        
        return output

    def backward(self, X: np.ndarray, outputs: np.ndarray,
                 targets: np.ndarray) -> None:
        """
        Args:
            X: images of shape [batch size, 785]
            outputs: outputs of model of shape: [batch size, num_outputs]
            targets: labels/targets of each image of shape: [batch size, num_classes]
        """
        assert targets.shape == outputs.shape,\
            f"Output shape: {outputs.shape}, targets: {targets.shape}"
        # A list of gradients.
        # For example, self.grads[0] will be the gradient for the first hidden layer
        self.grads = []
        delta_k = -(targets-outputs)    #delta_k.shape = [batch size, num_outputs]
        batch_size = targets.shape[0]

        delta_layer = delta_k
        self.grads.append((1/batch_size)*self.a_j[-1].T@delta_k)
                           
        for layer in range((len(self.neurons_per_layer)-1),0,-1):
            if self.use_improved_sigmoid == True:
                delta_layer = improved_sigmoid_derivative(self.a_j[layer-1],self.ws[layer-1])*(delta_layer@self.ws[layer].T)
            else:
                delta_layer = sigmoid_derivative(self.a_j[layer-1],self.ws[layer-1])*(delta_layer@self.ws[layer].T)
                           
            self.grads.insert(0,(1/batch_size)*(delta_layer.T@self.a_j[layer-1]).T)


        for grad, w in zip(self.grads, self.ws):
            assert grad.shape == w.shape,\
                f"Expected the same shape. Grad shape: {grad.shape}, w: {w.shape}."

    def zero_grad(self) -> None:
        self.grads = [None for i in range(len(self.ws))]


def one_hot_encode(Y: np.ndarray, num_classes: int):
    """
    Args:
        Y: shape [Num examples, 1]
        num_classes: Number of classes to use for one-hot encoding
    Returns:
        Y: shape [Num examples, num classes]
    """
    Yout = np.zeros((Y.shape[0],num_classes))
    for i in range(0,Y.shape[0]):
        Yout[i,Y[i]] = 1
    
    return Yout
    raise NotImplementedError


def gradient_approximation_test(
        model: SoftmaxModel, X: np.ndarray, Y: np.ndarray):
    """
        Numerical approximation for gradients. Should not be edited. 
        Details about this test is given in the appendix in the assignment.
    """
    epsilon = 1e-3
    for layer_idx, w in enumerate(model.ws):
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                orig = model.ws[layer_idx][i, j].copy()
                model.ws[layer_idx][i, j] = orig + epsilon
                logits = model.forward(X)
                cost1 = cross_entropy_loss(Y, logits)
                model.ws[layer_idx][i, j] = orig - epsilon
                logits = model.forward(X)
                cost2 = cross_entropy_loss(Y, logits)
                gradient_approximation = (cost1 - cost2) / (2 * epsilon)
                model.ws[layer_idx][i, j] = orig
                # Actual gradient
                logits = model.forward(X)
                model.backward(X, logits, Y)
                difference = gradient_approximation - \
                    model.grads[layer_idx][i, j]
                assert abs(difference) <= epsilon**2,\
                    f"Calculated gradient is incorrect. " \
                    f"Layer IDX = {layer_idx}, i={i}, j={j}.\n" \
                    f"Approximation: {gradient_approximation}, actual gradient: {model.grads[layer_idx][i, j]}\n" \
                    f"If this test fails there could be errors in your cross entropy loss function, " \
                    f"forward function or backward function"


if __name__ == "__main__":
    # Simple test on one-hot encoding
    Y = np.zeros((1, 1), dtype=int)
    Y[0, 0] = 3
    Y = one_hot_encode(Y, 10)
    assert Y[0, 3] == 1 and Y.sum() == 1, \
        f"Expected the vector to be [0,0,0,1,0,0,0,0,0,0], but got {Y}"

    X_train, Y_train, *_ = utils.load_full_mnist(0.1)
    X_train = pre_process_images(X_train)
    Y_train = one_hot_encode(Y_train, 10)
    assert X_train.shape[1] == 785,\
        f"Expected X_train to have 785 elements per image. Shape was: {X_train.shape}"

    neurons_per_layer = [64, 10]
    a_j = []
    use_improved_sigmoid = False
    use_improved_weight_init = False
    model = SoftmaxModel(
        neurons_per_layer, use_improved_sigmoid, use_improved_weight_init,a_j)

    # Gradient approximation check for 100 images
    X_train = X_train[:100]
    Y_train = Y_train[:100]
    for layer_idx, w in enumerate(model.ws):
        model.ws[layer_idx] = np.random.uniform(-1, 1, size=w.shape)

    gradient_approximation_test(model, X_train, Y_train)
