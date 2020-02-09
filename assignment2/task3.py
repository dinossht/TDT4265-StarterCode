import numpy as np
import utils
import matplotlib.pyplot as plt
import typing
from task2a import cross_entropy_loss, SoftmaxModel, one_hot_encode, pre_process_images
from sklearn.utils import shuffle
np.random.seed(0)

# TODO: Early stopping in the training loop. This is not required; 
# however, early stopping might enable you to stop training early 
# and save computation time Early stopping.
# NOTE: For each task we have set the hyperparameters (learning 
# rate and batch size) that should work fine for these tasks. 
# If you decide to change them, please state it in your report.


def calculate_accuracy(X: np.ndarray, targets: np.ndarray,
                       model: SoftmaxModel) -> float:
    """
    Args:
        X: images of shape [batch size, 785]
        targets: labels/targets of each image of shape: [batch size, 10]
        model: model of class SoftmaxModel
    Returns:
        Accuracy (float)
    """
    tot_preds = X.shape[0]  # total number of predictions
    currectly_predicted = np.sum(np.argmax(model.forward(X), 1)==np.argmax(targets, 1)) 
    accuracy = currectly_predicted / tot_preds 
    return accuracy


def train(
        model: SoftmaxModel,
        datasets: typing.List[np.ndarray],
        num_epochs: int,
        learning_rate: float,
        batch_size: int,
        # Task 3 hyperparameters,
        use_shuffle: bool,
        use_momentum: bool,
        momentum_gamma: float):
    X_train, Y_train, X_val, Y_val, X_test, Y_test = datasets

    # Utility variables
    num_batches_per_epoch = X_train.shape[0] // batch_size
    num_steps_per_val = num_batches_per_epoch // 5
    # Tracking variables to track loss / accuracy
    train_loss = {}
    val_loss = {}
    train_accuracy = {}
    val_accuracy = {}

    global_step = 0
    for epoch in range(num_epochs):
        for step in range(num_batches_per_epoch):
            start = step * batch_size
            end = start + batch_size
            X_batch, Y_batch = X_train[start:end], Y_train[start:end]

            # The mini-batch gradient descent algorithm
            model.backward(X_batch, model.forward(X_batch), Y_batch)
            model.ws[1] = model.ws[1] - learning_rate * model.grads[1]
            model.ws[0] = model.ws[0] - learning_rate * model.grads[0]

            # Track train / validation loss / accuracy
            # every time we progress 20% through the dataset
            if (global_step % num_steps_per_val) == 0:
                _train_loss = cross_entropy_loss(Y_train, model.forward(X_train))
                train_loss[global_step] = _train_loss

                _val_loss = cross_entropy_loss(Y_val, model.forward(X_val))
                val_loss[global_step] = _val_loss

                train_accuracy[global_step] = calculate_accuracy(
                    X_train, Y_train, model)
                val_accuracy[global_step] = calculate_accuracy(
                    X_val, Y_val, model)

            global_step += 1

        if use_shuffle:
            X_train, Y_train = shuffle(X_train, Y_train)


    return model, train_loss, val_loss, train_accuracy, val_accuracy


if __name__ == "__main__":
    # Load dataset
    validation_percentage = 0.2
    X_train, Y_train, X_val, Y_val, X_test, Y_test = utils.load_full_mnist(
        validation_percentage)

    # One hot encode data
    Y_train = one_hot_encode(Y_train, 10)
    Y_val   = one_hot_encode(Y_val, 10)
    Y_test  = one_hot_encode(Y_test, 10)

    # Preprocess data using mean and std of training set
    # Find mean and std of training set
    mu      = np.mean(X_train)
    sigma   = np.std(X_train)
    X_train = pre_process_images(X_train, mean=mu, std=sigma)
    X_val   = pre_process_images(X_val, mean=mu, std=sigma)    
    X_test  = pre_process_images(X_test, mean=mu, std=sigma)    

    # Hyperparameters
    num_epochs = 20
    learning_rate = .1
    batch_size = 32
    neurons_per_layer = [64, 10]
    momentum_gamma = .9  # Task 3 hyperparameter

    # Initialize variables for saving plotting data
    train_loss_all      = []
    val_loss_all        = []
    train_accuracy_all  = []
    val_accuracy_all    = []

    num_runs = 2
    for run in range(num_runs):
        print("Run:", run)
        
        # Settings for task 3. Keep all to false for task 2.
        use_shuffle                 = False
        use_improved_sigmoid        = False
        use_improved_weight_init    = False
        use_momentum                = False

        if run == 1:
            #use_shuffle = True
            use_improved_sigmoid        = True
            use_improved_weight_init    = True

        model = SoftmaxModel(
            neurons_per_layer,
            use_improved_sigmoid,
            use_improved_weight_init)
        
        model, train_loss, val_loss, train_accuracy, val_accuracy = train(
            model,
            [X_train, Y_train, X_val, Y_val, X_test, Y_test],
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            use_shuffle=use_shuffle,
            use_momentum=use_momentum,
            momentum_gamma=momentum_gamma)

        print("Final Train Cross Entropy Loss:",
            cross_entropy_loss(Y_train, model.forward(X_train)))
        print("Final Validation Cross Entropy Loss:",
            cross_entropy_loss(Y_val, model.forward(X_val)))
        print("Final Test Cross Entropy Loss:",
            cross_entropy_loss(Y_test, model.forward(X_test)))

        print("Final Train accuracy:",
            calculate_accuracy(X_train, Y_train, model))
        print("Final Validation accuracy:",
            calculate_accuracy(X_val, Y_val, model))
        print("Final Test accuracy:",
            calculate_accuracy(X_test, Y_test, model))

        # Save plot data
        train_loss_all.append(train_loss)
        val_loss_all.append(val_loss)
        train_accuracy_all.append(train_accuracy)
        val_accuracy_all.append(val_accuracy)


    fmt = ['-', '+']
    plt.figure(figsize=(20, 8))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.ylim([0, .4])
    
    for run in range(num_runs):
        global_steps = list(train_loss_all[run].keys())
        loss = list(train_loss_all[run].values())
        plt.plot(global_steps, loss, fmt[run])      
        
        global_steps = list(val_loss_all[run].keys())
        loss = list(val_loss_all[run].values())
        plt.plot(global_steps, loss, fmt[run])

    plt.xlabel("Number of gradient steps")
    plt.ylabel("Cross Entropy Loss")
    plt.legend(["Training Loss", "Validation Loss", "Training Loss with improved weight init", "Validation Loss with improved weight init"])
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.ylim([0.8, 1.0])
    for run in range(num_runs):
        global_steps = list(train_accuracy_all[run].keys())
        loss = list(train_accuracy_all[run].values())
        plt.plot(global_steps, loss, fmt[run])

        global_steps = list(val_accuracy_all[run].keys())
        loss = list(val_accuracy_all[run].values())
        plt.plot(global_steps, loss, fmt[run])

    plt.legend(["Training Accuracy", "Validation Accuracy", "Training Accuracy with improved weight init", "Validation Accuracy with improved weight init"])
    plt.xlabel("Number of gradient steps")
    plt.ylabel("Accuracy")
    
    # Save and show image
    plt.savefig("task3c_softmax_train_graph.png")
    plt.show()
