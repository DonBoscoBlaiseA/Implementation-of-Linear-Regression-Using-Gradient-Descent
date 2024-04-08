# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import StandardScaler to standardize the features and target variable.
2. Implement a function for linear regression with gradient descent, taking input features X1, target variable y, learning rate, and number of iterations as parameters.
3. Add a column of ones to feature matrix X for the intercept term and initialize theta (parameters) with zeros.
4. Iterate through the specified number of iterations, computing predictions, errors, and updating theta using gradient descent.
5. Read the dataset into a DataFrame, assuming the last column as the target variable 'y' and preceding columns as features 'X'.
6. Standardize the features and target variable using StandardScaler, learn model parameters using linear_regression function, and predict the target value for a new data point after scaling it.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Don Bosco Blaise A
RegisterNumber: 212221040045
*/
```
```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.1,num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1] #'_' is for concatenation
    theta=np.zeros(X.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        #calculate predictions
        predictions=(X).dot(theta).reshape(-1,1)
        #calculate errors
        errors=(predictions-y).reshape(-1,1)
        #update theta using gradient descent
        theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
    #theta is w 'hypothesis'
data=pd.read_csv("G:/jupyter_notebook_files/lab_ex_3/50_Startups.csv",header=None)
data.head()
#Assuming the last column is your target variable 'y' and the preceding columns
X=(data.iloc[1:,:-2].values)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X)
print(X1_Scaled)
#learn model parameters
```

```
theta=linear_regression(X1_Scaled,Y1_Scaled)
#predict target value for a new data point
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")
```

## Output:
![Screenshot (207)](https://github.com/DonBoscoBlaiseA/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/140850829/cb93aba5-4ac8-440a-b561-486e6b6eb5e6)
![Screenshot (208)](https://github.com/DonBoscoBlaiseA/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/140850829/50cfb905-8f1d-4f21-a1f1-dabcdd0cf717)
![Screenshot (209)](https://github.com/DonBoscoBlaiseA/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/140850829/845a8321-5e53-429e-aca7-a27f834939c8)
![Screenshot (210)](https://github.com/DonBoscoBlaiseA/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/140850829/8a35d235-d59e-42f3-a65a-9559c1a2a122)
![Screenshot (211)](https://github.com/DonBoscoBlaiseA/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/140850829/f9f0bf59-bca4-439e-8803-085845ceea3c)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
