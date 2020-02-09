import numpy as np
import utils
import typing
np.random.seed(1)


def sigmoid(x, improved_sigmoid=False):
    if improved_sigmoid:
        return 1.7159 * np.tanh((2.0 / 3.0) * x)

    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x, improved_sigmoid=False):
    if improved_sigmoid:
        return 1.14393 / np.cosh((2.0 / 3.0) * x)**2

    return sigmoid(x) * (1.0 - sigmoid(x))  


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
    # Not normalize by K! It will lead to gradients being much smaller
    ce = targets * np.log(outputs)
    N = targets.shape[0] 
    return (-1.0 / N) * np.sum(np.sum(ce)) 

class SoftmaxModel:

    def __init__(self,
                 # Number of neurons per layer
                 neurons_per_layer: typing.List[int],
                 use_improved_sigmoid: bool,  # Task 3a hyperparameter
                 use_improved_weight_init: bool  # Task 3c hyperparameter
                 ):
        # Define number of input nodes
        self.I = 785
        self.use_improved_sigmoid = use_improved_sigmoid

        # Define number of output nodes
        # neurons_per_layer = [64, 10] indicates that we will have two layers:
        # A hidden layer with 64 neurons and a output layer with 10 neurons.
        self.neurons_per_layer = neurons_per_layer

        # Initial the weight to randomly sampled weights from normal 
        # distribution with zero mean and standard deviation of 1/sqrt(fan-in)
        self.ws = []
        prev = self.I
        for size in self.neurons_per_layer:
            w_shape = (prev, size)
            print("Initializing weight to shape:", w_shape)
            # Initialize the weight to randomly sampled weight between [-1, 1]
            w = np.random.uniform(-1, 1, size=w.shape)
            
            if self.use_improved_sigmoid:
                # Fan-in standard deviation
                sigma = 1.0 / np.sqrt(size)
                w = np.random.normal(loc=0, scale=sigma, size=w_shape)
            
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
        # Sigmoid activation for hidden layer
        hidden_layer = sigmoid(X@self.ws[0], improved_sigmoid=self.use_improved_sigmoid)
        # Softmax for output layer
        y_hat = np.exp(hidden_layer@self.ws[1])
        return y_hat / np.sum(y_hat, axis=1, keepdims=True)

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
        
        # Equation 9 from assignmwent 1, but for two layers
        # Equations calculated in task 1b)
        # Backpropogate first from output to hidden, and then from hidden to input layer
        self.grads = []
        w1 = self.ws[0]
        w2 = self.ws[1]
        N = targets.shape[0]        
        
        # Output layer backpropogation
        delta_k = -(targets - outputs)
        z_j = X@w1
        a_j = sigmoid(z_j, improved_sigmoid=self.use_improved_sigmoid)
        dC_dw2 = (1.0 / N) * a_j.T@delta_k  
        
        # Hidden layer backpropogation
        delta_j = sigmoid_derivative(z_j, improved_sigmoid=self.use_improved_sigmoid) * (delta_k@w2.T)
        dC_dw1 = (1.0 / N) * X.T@delta_j 

        # Update gradient
        self.grads.append(dC_dw1)
        self.grads.append(dC_dw2)

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
    encoded_Y = np.zeros((len(Y), num_classes))  
    for i in range(len(Y)):
        encoded_Y[i, int(Y[i])] = 1
    return encoded_Y

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
    use_improved_sigmoid = False
    use_improved_weight_init = False
    model = SoftmaxModel(
        neurons_per_layer, use_improved_sigmoid, use_improved_weight_init)

    # Gradient approximation check for 100 images
    X_train = X_train[:100]
    Y_train = Y_train[:100]
    for layer_idx, w in enumerate(model.ws):
        model.ws[layer_idx] = np.random.uniform(-1, 1, size=w.shape)

    gradient_approximation_test(model, X_train, Y_train)
