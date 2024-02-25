import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as plt
import MyLinearRegression
import DataTransformation

X, y = DataTransformation.get_parameters()

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, train_size = 0.80)  # Going for a 80-20 split

# My Linear Regression #

m = x_train.size #Number of training examples
num_attributes = x_test.shape[1]
theta_initial = np.zeros((num_attributes, 1))  # Initialize parameters to zeros

X_train_scaled, mean_train, std_train = MyLinearRegression.feature_scaling(x_train)
X_test_scaled = (x_test - mean_train) / std_train  # Now all the examples are scaled
y_train = y_train.reshape(y_train.size, 1)

theta_final, cost_history = MyLinearRegression.gradient_descent(X_train_scaled, y_train, theta_initial,
                                                                learning_rate=0.05, num_iterations=100000)

y_pred_test = np.dot(X_test_scaled, theta_final)  # Making the predictions

mse, rmse = MyLinearRegression.evaluate_model(y_test, y_pred_test)

print("Testing MSE My Model:", mse)
print("Testing RMSE My Model:", rmse)

mae_test = MyLinearRegression.mean_absolute_error(y_test, y_pred_test)
print("Testing MAE My Model:", mae_test, '\n')

theta_final = theta_final.reshape(theta_final.size, )
print('Coefficients My Model: \n', theta_final, '\n')

features = ['Gender', 'Age', 'Height', 'Weight', 'Family History', 'High caloric\nfood frequently', 'Smoking',
            'Water intake', 'Monitor Calories', 'Physical activity', 'Alcohol']

plt.figure(figsize=(10, 6))
plt.barh(features, theta_final)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('My Linear Regression Model')

plt.show()

# THE LINEAR MODEL FROM SKLEARN #

linear = linear_model.LinearRegression()

linear.fit(x_train, y_train)

y_pred_test = np.dot(X_test_scaled, linear.coef_.T)  # Making the predictions

mse, rmse = MyLinearRegression.evaluate_model(y_test, y_pred_test)

print("Testing MSE Sklearn:", mse)
print("Testing RMSE Sklearn:", rmse)

mae_test = MyLinearRegression.mean_absolute_error(y_test, y_pred_test)
print("Testing MAE Sklearn:", mae_test, '\n')

print('Coefficient Sklearn: \n', linear.coef_)

linear_coef = linear.coef_.reshape(linear.coef_.size, )

plt.figure(figsize=(10, 6))
plt.barh(features, linear_coef)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Sklearn Regression Model')

plt.show()



