# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the California Housing dataset and select the first 3 features as input (X) and target variables (Y) (including the target price and another feature).
2. Split the data into training and testing sets, then scale (standardize) both the input features and target variables.
3. Train a multi-output regression model using Stochastic Gradient Descent (SGD) on the training data.
4. Make predictions on the test data, inverse transform the predictions, calculate the Mean Squared Error, and print the results.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Pradeep E
RegisterNumber:  212223230149
*/
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

data=fetch_california_housing()
X=data.data[:, :3]
Y=np.column_stack((data.target, data.data[:, 6]))

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

scaler_X=StandardScaler()
scaler_Y=StandardScaler()

X_train=scaler_X.fit_transform(X_train)
X_test=scaler_X.transform(X_test)
Y_train=scaler_Y.fit_transform(Y_train)
Y_test=scaler_Y.transform(Y_test)

sgd=SGDRegressor(max_iter=1000,tol=1e-3)
multi_output_sgd=MultiOutputRegressor(sgd)
multi_output_sgd.fit(X_train,Y_train)
Y_pred=multi_output_sgd.predict(X_test)

Y_pred=scaler_Y.inverse_transform(Y_pred)
Y_test=scaler_Y.inverse_transform(Y_test)
mse=mean_squared_error(Y_test,Y_pred)
print("Mean Squared Error:",mse)
print("\nPredictions:\n",Y_pred[:5])
```

## Output:
![the Logistic Regression Model to Predict the Placement Status of Student](sam.png)

![Screenshot 2024-09-01 172631](https://github.com/user-attachments/assets/6656c4b3-7ef3-45d6-911e-547d5850035b)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
