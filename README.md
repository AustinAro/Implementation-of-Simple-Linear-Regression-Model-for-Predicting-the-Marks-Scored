# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries. 
2.Set variables for assigning dataset values. 
3.Import linear regression from sklearn. 
4.Assign the points for representing in the graph. 
5.Predict the regression for marks by using the representation of the graph. 
6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Austin Aro A
RegisterNumber:  212224040038
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:
# DATASET :

![Screenshot 2025-04-16 162134](https://github.com/user-attachments/assets/689bf889-c90f-43f6-88a2-6d4fe8e2fc8e)

# HEAD VALUE :

![Screenshot 2025-04-16 162151](https://github.com/user-attachments/assets/ca4c7682-fe84-4a5e-9d34-6bcaaae7cd01)

# TAIL VALUE :

![Screenshot 2025-04-16 162155](https://github.com/user-attachments/assets/693576e8-b289-4cbc-9fc4-777739a950a1)

# X AND Y VALUE :

![Screenshot 2025-04-16 162218](https://github.com/user-attachments/assets/45f232ae-d974-4595-9018-4a1a806a746c)

# PREDICATION OF VALUE X AND Y :

![Screenshot 2025-04-16 162239](https://github.com/user-attachments/assets/55f92b8d-2a19-4a32-80d3-3ac24a6e5055)

# MSE,MAE,RMSE :

![Screenshot 2025-04-16 162259](https://github.com/user-attachments/assets/768b329b-da12-4b5f-ab3a-8c78ec280199)

# TRAINING SET :

![Screenshot 2025-04-16 162250](https://github.com/user-attachments/assets/fb64f886-970c-49df-bd75-edd671b1d980)

# TESTING SET :

![Screenshot 2025-04-16 162255](https://github.com/user-attachments/assets/5c68a56b-9191-486e-bb70-d3e8fe0437ae)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
