# no imports beyond the ones below should be needed in answering this question
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def sigmoid(z):
	"""sigmoid function"""
	result = 1 / (1 + np.exp(-z))

	assert np.all(0.0 <= result)

	assert np.all(result <= 1.0)

	return result

def log_loss(y, y_probs):
	"""log-loss function"""
	assert np.all(y_probs >= 0.0)

	assert np.all(y_probs <= 1.0)

	result = (-y * np.log(y_probs) - (1 - y) * np.log(1 - y_probs)).mean()

	return result 

def log_loss_grad(X, y, w):
	"""gradient of the log-loss function"""
	assert len(X.shape)==2

	assert len(y.shape)==1

	assert X.shape[1] == w.shape[0]

	assert X.shape[0] == y.shape[0]

	y_probs = predict_probs(X, w)

	assert y_probs.shape[0] == y.shape[0]

	result = np.dot(X.T, (y_probs - y)) / y.shape[0]

	return result

def predict_probs(X, w):
	"""predict logistic regression probabilities"""
	assert X.shape[1] == w.shape[0]

	result = sigmoid(np.dot(X, w))

	return result

def predict(X, w, threshold=0.5):
	"""make logistic regression predictions using a specified threshold
	to binarize the probability threshold
	"""
	return 1.0 * (predict_probs(X, w) >= threshold)

def evaluate_accuracy(X, y, w):
	"""evaluate accuracy by making predictions
	and comparing with groundtruth"""
	y_predict = predict(X, w)

	result = (y == y_predict).mean()

	assert (0.0 <= result <= 1.0)

	return result

def gradient_descent(w, X, y, f_grad, lr = 1e-2):
	"""makes an update using gradient descent
	where the gradient is calculated using all the data
	
	Parameters:
		w: current weight parameter, shape = num_features
		X: input features, shape = num_datapoints, num_features
		y: binary output target, shape = num_datapoints
		f_grad: a Python function which computes the gradient of the log-loss function, use log_loss_grad here
		lr: learning rate for gradient descent
	"""
	# Do not edit any code outside the edit region
	# Edit region starts here
	#########################
	# Your code goes here
	grad = f_grad(X, y, w)  # Compute the gradient of the log-loss
	w -= lr * grad
	#########################
	# Edit region ends here

	assert X.shape[1] == w.shape[0]

def stochastic_gradient_descent(w, X, y, f_grad, lr = 1e-2):
	"""makes an update using stochastic gradient descent
	where the gradient is calculated using a randomly chosen datapoint

	Parameters:
		w: current weight parameter, shape = num_features
		X: input features, shape = num_datapoints, num_features
		y: binary output target, shape = num_datapoints
		f_grad: a Python function which computes the gradient of the log-loss function, use log_loss_grad here
		lr: learning rate for stochastic gradient descent
	"""
	# Do not edit any code outside the edit region
	# Edit region starts here
	#########################
	# Your code goes here
	i = np.random.randint(0, X.shape[0])  # Select a random index
	grad = f_grad(X[i:i+1], y[i:i+1], w)  # Compute the gradient for the selected datapoint
	w -= lr * grad
	#########################
	# Edit region ends here

	assert X.shape[1] == w.shape[0]

def adagrad(w, X, y, f_grad, gti, lr = 1e-2, eps_stable = 1e-8):
	"""makes an update using adagrad
	where the gradient is calculated using a randomly chosen datapoint
	and gti maintains the running sum of squared gradient magnitudes required for the adagrad update

	Parameters:
		w: current weight parameter, shape = num_features
		X: input features, shape = num_datapoints, num_features
		y: binary output target, shape = num_datapoints
		f_grad: a Python function which computes the gradient of the log-loss function, use log_loss_grad here
		gti: maintains the running sum of squared gradient magnitude, shape = num_features
		lr: learning rate for AdaGrad
	"""
	# Do not edit any code outside the edit region
	# Edit region starts here
	#########################
	# Your code goes here
	grad = f_grad(X, y, w)  # Compute the gradient of the log-loss
	gti += grad ** 2  # Accumulate squared gradients
	w -= lr * grad / (np.sqrt(gti) + eps_stable)
	#########################
	# Edit region ends here

	assert X.shape[1] == w.shape[0]

if __name__ == '__main__':
    # Set numpy seed for reproducibility
    np.random.seed(666)

    # Load well-known Iris dataset from scikit-learn package
    # Convert from 3 to 2 classes for binary classification
    iris = load_iris()
    X = iris.data[:, :2]
    y = (iris.target != 0) * 1

    # Add a constant feature at position 0 of datapoints
    # The first weight therefore corresponds to the bias term
    intercept = np.ones((X.shape[0], 1))
    X = np.concatenate((intercept, X), axis=1)

    # Split data into train and test using a scikit-learn utility function
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=666)

    # Number of epochs
    num_epochs = 10000

    # Initialize weights for gradient descent, stochastic gradient descent, adagrad
    w_gd = np.zeros(X.shape[1])
    w_sgd = np.zeros(X.shape[1])
    w_adagrad = np.zeros(X.shape[1])

    # Initialize G_tii for adagrad
    gti = np.zeros(X.shape[1])

    # Create lists to store the trajectories for w1, w2 during training
    w_gd_trajectory = []
    w_sgd_trajectory = []
    w_adagrad_trajectory = []

    # Lists for storing logistic loss and accuracy during training
    train_logloss_gd = []
    train_logloss_sgd = []
    train_logloss_adagrad = []

    train_accuracy_gd = []
    train_accuracy_sgd = []
    train_accuracy_adagrad = []

    for epoch in range(num_epochs):
        # Gradient Descent update
        gradient_descent(w_gd, X_train, y_train, log_loss_grad)
        w_gd_trajectory.append(w_gd[1:3])  # Track w1, w2 for GD
        train_logloss_gd.append(log_loss(y_train, predict_probs(X_train, w_gd)))
        train_accuracy_gd.append(evaluate_accuracy(X_train, y_train, w_gd))

        # Stochastic Gradient Descent and AdaGrad updates
        for i in range(X_train.shape[0]):
            stochastic_gradient_descent(w_sgd, X_train, y_train, log_loss_grad)
            adagrad(w_adagrad, X_train, y_train, log_loss_grad, gti)

        # Track w1, w2 for SGD and AdaGrad
        w_sgd_trajectory.append(w_sgd[1:3])
        w_adagrad_trajectory.append(w_adagrad[1:3])

        # Track loss and accuracy for SGD and AdaGrad
        train_logloss_sgd.append(log_loss(y_train, predict_probs(X_train, w_sgd)))
        train_accuracy_sgd.append(evaluate_accuracy(X_train, y_train, w_sgd))

        train_logloss_adagrad.append(log_loss(y_train, predict_probs(X_train, w_adagrad)))
        train_accuracy_adagrad.append(evaluate_accuracy(X_train, y_train, w_adagrad))

    # Convert trajectories to numpy arrays for easy plotting
    w_gd_trajectory = np.array(w_gd_trajectory)
    w_sgd_trajectory = np.array(w_sgd_trajectory)
    w_adagrad_trajectory = np.array(w_adagrad_trajectory)

    # Plot Weight Parameter Trajectories
    plt.figure(figsize=(8, 6))
    plt.plot(w_gd_trajectory[:, 0], w_gd_trajectory[:, 1], label='Gradient Descent', color='blue', marker='o')
    plt.plot(w_sgd_trajectory[:, 0], w_sgd_trajectory[:, 1], label='Stochastic Gradient Descent', color='red', marker='x')
    plt.plot(w_adagrad_trajectory[:, 0], w_adagrad_trajectory[:, 1], label='AdaGrad', color='green', marker='^')

    plt.xlabel('w1')
    plt.ylabel('w2')
    plt.title('Weight Parameter Trajectories')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Create a meshgrid covering the input feature space
    xx, yy = np.meshgrid(np.linspace(X_train[:, 1].min(), X_train[:, 1].max(), 100),
                         np.linspace(X_train[:, 2].min(), X_train[:, 2].max(), 100))

    # Predict probabilities for the mesh grid
    grid_points = np.c_[np.ones(xx.ravel().shape[0]), xx.ravel(), yy.ravel()]
    prob_gd = predict_probs(grid_points, w_gd).reshape(xx.shape)
    prob_sgd = predict_probs(grid_points, w_sgd).reshape(xx.shape)
    prob_adagrad = predict_probs(grid_points, w_adagrad).reshape(xx.shape)

    # Plot Decision Boundaries and Data Points
    plt.figure(figsize=(10, 8))

    # Plot decision boundaries for each optimizer
    plt.contourf(xx, yy, prob_gd, levels=[0, 0.5, 1], colors=['lightblue', 'darkblue'], alpha=0.5)
    plt.contourf(xx, yy, prob_sgd, levels=[0, 0.5, 1], colors=['lightcoral', 'darkred'], alpha=0.5)
    plt.contourf(xx, yy, prob_adagrad, levels=[0, 0.5, 1], colors=['lightgreen', 'darkgreen'], alpha=0.5)

    # Plot training data points
    plt.scatter(X_train[y_train == 0, 1], X_train[y_train == 0, 2], color='red', marker='o', label='Train Negative')
    plt.scatter(X_train[y_train == 1, 1], X_train[y_train == 1, 2], color='green', marker='o', label='Train Positive')

    # Plot test data points
    plt.scatter(X_test[y_test == 0, 1], X_test[y_test == 0, 2], color='red', marker='x', label='Test Negative')
    plt.scatter(X_test[y_test == 1, 1], X_test[y_test == 1, 2], color='green', marker='x', label='Test Positive')

    # Add labels and legend
    plt.xlabel('Feature 1 (w1)')
    plt.ylabel('Feature 2 (w2)')
    plt.title('Data Points and Decision Boundaries')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot Logistic Loss vs Epoch
    plt.figure(figsize=(10, 6))
    plt.plot(range(num_epochs), train_logloss_gd, label='Gradient Descent', color='blue')
    plt.plot(range(num_epochs), train_logloss_sgd, label='Stochastic Gradient Descent', color='red')
    plt.plot(range(num_epochs), train_logloss_adagrad, label='AdaGrad', color='green')

    plt.xlabel('Epoch')
    plt.ylabel('Train Logistic Loss')
    plt.title('Train Logistic Loss vs Epoch')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot Accuracy vs Epoch
    plt.figure(figsize=(10, 6))
    plt.plot(range(num_epochs), train_accuracy_gd, label='Gradient Descent', color='blue')
    plt.plot(range(num_epochs), train_accuracy_sgd, label='Stochastic Gradient Descent', color='red')
    plt.plot(range(num_epochs), train_accuracy_adagrad, label='AdaGrad', color='green')

    plt.xlabel('Epoch')
    plt.ylabel('Train Accuracy')
    plt.title('Train Accuracy vs Epoch')
    plt.legend()
    plt.grid(True)
    plt.show()
	
test_accuracy_gd = evaluate_accuracy(X_test, y_test, w_gd)
print(f"Test Accuracy (Gradient Descent): {test_accuracy_gd * 100:.2f}%")

test_accuracy_sgd = evaluate_accuracy(X_test, y_test, w_sgd)
print(f"Test Accuracy (Stochastic Gradient Descent): {test_accuracy_sgd * 100:.2f}%")

test_accuracy_adagrad = evaluate_accuracy(X_test, y_test, w_adagrad)
print(f"Test Accuracy (AdaGrad): {test_accuracy_adagrad * 100:.2f}%")



