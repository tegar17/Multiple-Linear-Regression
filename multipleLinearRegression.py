"""A NumPy implementation of multiple linear regression using gradient descent.

The model learns one coefficient per input feature plus a bias term.
"""

from copy import error
import matplotlib.pyplot as plt
import numpy as np
import warnings 


class MultipleLinearRegression():
    """Fit and use a multiple linear regression model.

    Attributes:
        weights: Learned feature coefficients followed by the bias coefficient.
        loss_history: Mean squared error recorded during model training.
    """

    def __init__(self):
        """Initialize the model before fitting."""
        self.weights = None
        self.loss_history = []

    def fit_linear_regression(self, x, y, learning_rate=0.01, epochs=1000, print_interval=100):
        """Fit model weights to the supplied training data.

        Args:
            x: Two-dimensional array of training features.
            y: Target values corresponding to the rows in ``x``.
            learning_rate: Step size used by gradient descent.
            epochs: Maximum number of training iterations.
            print_interval: Number of epochs between progress messages.

        Returns:
            The learned coefficients, with the bias as the final value.
        """
        n = len(y)
        X_b = np.c_[x, np.ones(n)] # add bias term
        self.weights = np.zeros(X_b.shape[1])  # initialize weights
        y = y.flatten()
        self.loss_history = []

        for epoch in range(epochs):
            y_pred = X_b @ self.weights
            error = y_pred - y

            mse = (1 / n) * np.sum(np.square(error))
            self.loss_history.append(mse)

            if epoch % print_interval == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch:5d} | Loss (MSE): {mse:.4f}")
                      
            gradients = -2/n * X_b.T @ (y - y_pred)
            
            gradients = (2/n) * (X_b.T @ error)
            self.weights = self.weights - learning_rate * gradients

            if np.isnan(np.sum(self.weights)):
                warnings.warn('Try smaller learning rate')
                break
        return self.weights
        

    def predict(self, x):
        """Predict target values for one or more rows of input features.

        Args:
            x: Two-dimensional array containing the feature rows to predict.

        Returns:
            A NumPy array of predictions.
        """
        x_b = np.c_[x, np.ones((len(x)))]  # add bias term
        return x_b @ self.weights

    def plotLoss(self):
        """Display the mean squared error recorded for each training epoch."""
        epochs = range(len(self.loss_history))
        plt.plot(epochs, self.loss_history)
        plt.xlabel('Number of Epochs')
        plt.ylabel('Loss')
        plt.title('Plot Loss')
        plt.show()

    def train_test_split(self, X, y, test_size=0.2, random_state=42):
        """Shuffle and split feature and target arrays into train and test sets.

        Args:
            X: Feature array.
            y: Target array aligned row-for-row with ``X``.
            test_size: Fraction of rows assigned to the test set.
            random_state: Seed for reproducible shuffling, or ``None`` for
                non-deterministic shuffling.

        Returns:
            ``(X_train, X_test, y_train, y_test)`` in that order.
        """
        # Fix the random seed to make the split reproducible.
        if random_state is not None:
            np.random.seed(random_state)
            
        # Shuffle row indices so the split is not dependent on input order.
        shuffled_indices = np.random.permutation(len(X))
        
        # Reserve the requested fraction of rows for the test set.
        test_set_size = int(len(X) * test_size)
        
        test_indices = shuffled_indices[:test_set_size]
        train_indices = shuffled_indices[test_set_size:]
        
        # Apply the same row indices to features and targets.
        return X[train_indices], X[test_indices], y[train_indices], y[test_indices]