import numpy as np


def cost_function(X, y, theta):
    predictions = np.dot(X, theta)
    squared_error = (predictions - y) ** 2
    return np.mean(squared_error) / 2


# here is a second cost function I found online and is slightly better
def cost_function2(X, y, theta):
    predictions = np.dot(X, theta)
    squared_error = (predictions - y) ** 2
    return squared_error / (2 * y.size)


# Classic algorithm
def gradient_descent(X, y, theta, learning_rate, num_iterations):
    m = len(y)
    cost_history = []
    for _ in range(num_iterations):
        predictions = np.dot(X, theta)
        errors = predictions - y
        gradient = np.dot(X.T, errors) / m
        theta -= learning_rate * gradient
        cost = cost_function2(X, y, theta)
        cost_history.append(cost)
    return theta, cost_history


# Here I tried to find the best learning rate, and I found out that it does not change the error much
def find_best_learning_rate(X_train, X_test, y_train, y_test, theta, num_iterations):
    m = len(y_train)
    learning_rate_history = []
    mse_history = []
    learning_rate = 0.1
    while learning_rate < 1:
        theta_new, _ = gradient_descent(X_train, y_train, theta, learning_rate, num_iterations)
        learning_rate_history.append(learning_rate)

        y_pred_test = np.dot(X_test, theta_new)
        mae = mean_absolute_error(y_test, y_pred_test)
        mse_history.append(mae)

        learning_rate += 0.1
        theta = theta_new

    mini = mse_history.index(min(mse_history))
    print('Learning rate history: \n', learning_rate_history)
    print('Error history: \n', mse_history)

    return learning_rate_history[mini]


# Here are the results:
# Learning rate history:
#  [0.1, 0.2, 0.30000000000000004, 0.4, 0.5, 0.6, 0.7, 0.7999999999999999, 0.8999999999999999, 0.9999999999999999]
# Error history:
#  [3.430947864379368, 3.430947864379541, 3.430947864379568, 3.430947864379568, 3.430947864379568, 3.430947864379568, 3.430947864379568, 3.430947864379568, 3.430947864379568, nan]
# Almost no change :)


def feature_scaling(X):
    # Standardization (subtract mean and divide by standard deviation)
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_scaled = (X - mean) / std
    return X_scaled, mean, std


# Mean squared error and Root mean squared error
def evaluate_model(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    return mse, rmse


# Mean absolute error
def mean_absolute_error(y_true, y_pred):
    absolute_errors = np.abs(y_true - y_pred)
    mae = np.mean(absolute_errors)
    return mae
