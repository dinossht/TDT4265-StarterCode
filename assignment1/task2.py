import numpy as np
import utils
import sys 
import matplotlib.pyplot as plt
from task2a import cross_entropy_loss, BinaryModel, pre_process_images
np.random.seed(0)

INT_MAX = sys.maxsize 

def calculate_accuracy(X: np.ndarray, targets: np.ndarray, model: BinaryModel) -> float: 
    """
    Args:
        X: images of shape [batch size, 785]
        targets: labels/targets of each image of shape: [batch size, 1]
        model: model of class BinaryModel
    Returns:
        Accuracy (float)
    """
    # Task 2c
    tot_preds = X.shape[0]  # total number of predictions
    num_errors = np.sum(abs(targets-model.forward(X).round()))  # abs error between target and prediction
    accuracy = (tot_preds-num_errors)/tot_preds 
    return accuracy  


early_stopping_step = 0
def train(
        num_epochs: int,
        learning_rate: float,
        batch_size: int,
        l2_reg_lambda: float # Task 3 hyperparameter. Can be ignored before this.
        ):
    """
        Function that implements logistic regression through mini-batch
        gradient descent for the given hyperparameters
    """
    global X_train, X_val, X_test, early_stopping_step
    # Utility variables
    num_batches_per_epoch = X_train.shape[0] // batch_size
    num_steps_per_val = num_batches_per_epoch // 5
    train_loss = {}
    val_loss = {}
    train_accuracy = {}
    val_accuracy = {}
    model = BinaryModel(l2_reg_lambda)

    # Early stopping var init
    last_loss = INT_MAX 
    already_failed = 0

    global_step = 0
    for epoch in range(num_epochs):
        for step in range(num_batches_per_epoch):
            # Select our mini-batch of images / labels
            start = step * batch_size
            end = start + batch_size
            X_batch, Y_batch = X_train[start:end], Y_train[start:end]

            # The mini-batch gradient descent algorithm for m batches and a single epoch. 
            model.backward(X_batch,model.forward(X_batch),Y_batch)
            model.w = model.w-learning_rate*model.grad

            # Track training loss continuously
            _train_loss = cross_entropy_loss(Y_batch,model.forward(X_batch))
            train_loss[global_step] = _train_loss[0,0]
            
            # Track validation loss / accuracy every time we progress 20% through the dataset
            if global_step % num_steps_per_val == 0:
                _val_loss = cross_entropy_loss(Y_val,model.forward(X_val))
                val_loss[global_step] = _val_loss[0,0]

                train_accuracy[global_step] = calculate_accuracy(
                    X_train, Y_train, model)
                val_accuracy[global_step] = calculate_accuracy(
                    X_val, Y_val, model)

                # Early stopping criteria    
                if(_val_loss[0,0]>last_loss and already_failed>20):
                    # Stop early
                    #print("Early stopping kicked in at epoch nr.:",epoch+1)
                    #return model, train_loss, val_loss, train_accuracy, val_accuracy
                    if early_stopping_step == 0:
                        early_stopping_step = global_step 

                # Means failed this round
                elif(_val_loss[0,0]>last_loss): 
                    already_failed += 1

                # The loss improved this round, reset counter    
                else: 
                    last_loss = _val_loss[0,0] 
                    already_failed = 0

            global_step += 1
    return model, train_loss, val_loss, train_accuracy, val_accuracy


# Load dataset
category1, category2 = 2, 3
validation_percentage = 0.1
X_train, Y_train, X_val, Y_val, X_test, Y_test = utils.load_binary_dataset(
    category1, category2, validation_percentage)

# Preprocess dataset
X_train = pre_process_images(X_train)    
X_test  = pre_process_images(X_test)
X_val   = pre_process_images(X_val)

# hyperparameters
num_epochs = 50
learning_rate = 0.2
batch_size = 128
l2_reg_lambda = 0  # [0 1.0, 0.1, 0.01, 0.001]
model, train_loss, val_loss, train_accuracy, val_accuracy = train(
    num_epochs=num_epochs,
    learning_rate=learning_rate,
    batch_size=batch_size,
    l2_reg_lambda=l2_reg_lambda)

print("Final Train Cross Entropy Loss:",
      cross_entropy_loss(Y_train, model.forward(X_train)))
print("Final Validation Cross Entropy Loss:",
      cross_entropy_loss(Y_test, model.forward(X_test)))
print("Final Test Cross Entropy Loss:",
      cross_entropy_loss(Y_val, model.forward(X_val)))


print("Train accuracy:", calculate_accuracy(X_train, Y_train, model))
print("Validation accuracy:", calculate_accuracy(X_val, Y_val, model))
print("Test accuracy:", calculate_accuracy(X_test, Y_test, model))


# Plot loss
plt.ylim([0., .4]) 
utils.plot_loss(train_loss, "Training Loss")
utils.plot_loss(val_loss, "Validation Loss")
plt.vlines(early_stopping_step,0,1,label="Early stopping")

plt.legend()
plt.savefig("binary_train_loss.png")
plt.show()


# Plot accuracy
plt.ylim([0.93, .99])
utils.plot_loss(train_accuracy, "Training Accuracy")
utils.plot_loss(val_accuracy, "Validation Accuracy")
plt.vlines(early_stopping_step,0,1,label="Early stopping")

plt.legend()
plt.savefig("binary_train_accuracy.png")
plt.show()



# Loop through different lamda values
lamda = [1.0, 0.1, 0.01, 0.001]
len_w = [0,0,0,0]
for i in range(4):
    l2_reg_lambda = lamda[i]  
    model, train_loss, val_loss, train_accuracy, val_accuracy = train(
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        l2_reg_lambda=l2_reg_lambda)

    print("Validation accuracy:", calculate_accuracy(X_val, Y_val, model))
    
    plt.figure(10)
    utils.plot_loss(val_accuracy, "Validation Accuracy")

    # Plot the length of the weight vector (L2-norm) for each lamda
    len_w[i] = np.linalg.norm(model.w)

    # Reshape weights to image 2 d)
    plt.figure(12)
    plt.subplot(1,4,i+1)
    plt.imshow(np.reshape(np.array(model.w[0:28**2]),(28,28)))

plt.figure(10)    
plt.ylim([0.93, .99])
plt.legend(lamda)
plt.savefig("validatation_accuracy_different_lamdas.png")
plt.show()

plt.figure(11)
plt.plot(lamda,len_w)
plt.xlabel('lamda')
plt.ylabel('Length of weight vector')
plt.savefig("L2_norm_vs_lamda.png")
plt.show()

plt.figure(12)
plt.savefig('weight.png')
plt.show()
