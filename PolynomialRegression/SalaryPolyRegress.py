import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression


dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

#Training the linear regression model on the whole dataset
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 1/5,random_state=1 )
x_train = x_train.reshape(-1,1)
x_test = x_test.reshape(-1,1)
regressor = LinearRegression()
