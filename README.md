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
Developed by: 
RegisterNumber:  
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
![image](https://github.com/jeyaqbalan7/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119393851/efa51203-97bc-4cf9-b51a-7e06a0bf85df)

![image](https://github.com/jeyaqbalan7/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119393851/422b0146-3571-4184-8d8c-cce053e0ff6e)

![image](https://github.com/jeyaqbalan7/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119393851/7848292e-a217-4391-a82d-868bcb8c5bea)

![image](https://github.com/jeyaqbalan7/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119393851/f7c63faf-7cf5-48d0-83a4-323b1845d519)

![image](https://github.com/jeyaqbalan7/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119393851/f4e702ad-0f76-4852-af41-062b821cf2ac)

![image](https://github.com/jeyaqbalan7/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119393851/24110f4a-76df-414d-a813-d636bf01376b)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
