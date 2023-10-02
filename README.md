# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

Step 1: Use the standard libraries such as numpy, pandas, matplotlib.pyplot in python for the Gradient Descent.

Step 2: Upload the dataset conditions and check for any null value in the values provided using the .isnull() function.

Step 3: Declare the default values such as n, m, c, L for the implementation of linear regression using gradient descent.

Step 4: Calculate the loss using Mean Square Error formula and declare the variables y_pred, dm, dc to find the value of m.

Step 5: Predict the value of y and also print the values of m and c.

Step 6: Plot the accquired graph with respect to hours and scores using the scatter plot function.

Step 7: End the program.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Jeyabalan
RegisterNumber: 212222240040
*/
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv("/content/ex1.txt",header=None)

plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

def computeCost(x,y,theta):
  m=len(y)
  h=x.dot(theta)
  square_err=(h-y)**2
  return 1/(2*m)*np.sum(square_err)
  
data_n=data.values
m=data_n[:,0].size
x=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))

computeCost(x,y,theta)
def gradientDescent(x,y,theta,alpha,num_iters):
  m=len(y)
  J_history=[]

  for i in range(num_iters):
    predictions = x.dot(theta)
    error = np.dot(x.transpose(),(predictions -y))
    descent=alpha*1/m*error
    theta-=descent
    J_history.append(computeCost(x,y,theta))

  return theta,J_history
  theta,J_history = gradientDescent(x,y,theta,0.01,1500)
print("h(x) *"+str(round(theta[0,0],2))+"+"+str(round(theta[1,0],2))+"x1")

plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\theta)$")
plt.title("Cost frunction using Gradient Descent")

plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0]for y in x_value]
plt.plot(x_value,y_value,color="r")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit ($10,000)")
plt.ylabel("Profit predictions")

def predict(x,theta):
  predictions=np.dot(theta.transpose(),x)
  return predictions[0]

predict1=predict(np.array([1,3.5]),theta)*10000
print("For population = 35,000, we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("For population = 70,000, we predict a profit of $"+str(round(predict2,0)))
```

## Output:
Profit prediction:

![image](https://github.com/jeyaqbalan7/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119393851/01f36032-732f-4fd8-b7e0-259f90ff53a9)

Function output:


![image](https://github.com/jeyaqbalan7/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119393851/f9450973-8733-4fa4-a037-25aa384029d3)

Gradient descent:


![image](https://github.com/jeyaqbalan7/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119393851/2ae5c070-223c-4d31-a45e-d8b63c095b7b)

Cost function using Gradient Descent:

![image](https://github.com/jeyaqbalan7/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119393851/464a4a6c-e258-4b8e-b1be-e974e38d3c9f)

Linear Regression using Profit Prediction:

![image](https://github.com/jeyaqbalan7/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119393851/e855b43f-fd25-4161-892b-a9ce3a4f4ade)

Profit Prediction for a population of 35000:

![image](https://github.com/jeyaqbalan7/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119393851/e7d31c88-c41b-41da-8882-14c49bf58307)

Profit Prediction for a population of 70000 :

![image](https://github.com/jeyaqbalan7/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119393851/11717915-c7b8-4e8e-91d4-d91ee42106ae)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
