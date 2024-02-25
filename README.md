# Obesity Estimation with Linear Regression

This repository contains a custom implementation of a linear regression model for estimating obesity levels based on specific attributes. The model is compared to the linear regression model provided by Scikit-learn using an obesity dataset.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dependencies](#dependencies)
- [Data Visualization](#data-visualization)
- [Coeficients comparison](#coeficients-comparison)
- [Results](#results)
- [Possible Improvements](#possible-improvements)

## Overview
- The dataset used is Estimation of obesity levels based on eating habits and physical condition, which can be found <a href="https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition">here</a>. 
 More information regarding the dataset are found <a href="https://www.sciencedirect.com/science/article/pii/S2352340919306985?via%3Dihub">here</a>.
- The features used from the dataset are: Gender, Age, Height, Weight, Family history, Frequency of high caloric food, Smoking, Water intake, Monitoring calories, Physical activity and Alcohol consumption.
- This custom implementation of linear regression includes the gradient descent optimization algorithm, which iteratively updates model parameters to minimize the cost function and improve predictive performance. It also includes a function for finding the best learning rate.


## Features
- Custom implementation of linear regression
- Feature engineering and preprocessing
- Model comparison with Scikit-learn's linear regression
- Data visualization and analysis

## Dependencies
- Python 3.x
- NumPy
- pandas
- Matplotlib
- Scikit-learn

## Data visualization
![data_visiualisation](https://github.com/RazvanGolan/ObesityPredictor/assets/117024228/59857b89-93f2-4f7f-ac03-247f79f3e212)

## Coeficients comparison
<img width="991" alt="Screenshot 2024-02-25 at 16 50 38" src="https://github.com/RazvanGolan/ObesityPredictor/assets/117024228/3dcb3553-339f-4697-afca-31f2154eab2c">


<img width="997" alt="Screenshot 2024-02-25 at 15 46 24" src="https://github.com/RazvanGolan/ObesityPredictor/assets/117024228/798226bb-4c92-42d6-8d46-79db3fcf375f">

### Analysis of Feature 4 Coefficient - Weight
- The coefficient for Feature 4 in the custom model is substantially higher compared to the coefficient obtained from Scikit-learn's model.
- This indicates that in the custom model, Feature 4 has a much stronger positive impact on the predicted obesity levels.
- It suggests that in the context of the custom model, changes in Feature 4 have a greater influence on the predicted obesity levels compared to other features.

### Possible Explanations:
- Feature Engineering Differences: it's possible that there are differences in how Feature 4 is engineered or preprocessed in the custom model compared to Scikit-learn's model. Differences in scaling, normalization, or encoding methods could lead to variations in the coefficient values.
- Modeling Assumptions: the custom model may make different assumptions or have different underlying mathematical formulations compared to Scikit-learn's model, leading to variations in coefficient values.
- Overfitting or Underfitting: differences in coefficient values could also be attributed to overfitting or underfitting of the models. The custom model may be overfitting to the training data, resulting in inflated coefficient values for certain features.

## Results
The custom linear regression model achieved comparable results to Scikit-learn's linear regression model on the obesity dataset. The model's performance was evaluated using metrics such as Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE).

<img width="671" alt="Screenshot 2024-02-25 at 16 53 39" src="https://github.com/RazvanGolan/ObesityPredictor/assets/117024228/c3c763d2-c8b4-4361-99b6-8126a3cfddf1">


## Possible Improvements
- Explore more advanced feature selection methods
- Handling Outliers and Missing Data More Effectively
- Regularization and Model Complexity Control
- Incorporating Domain Knowledge or Additional Data Sources
- Hyperparameter Tuning for Model Optimization

