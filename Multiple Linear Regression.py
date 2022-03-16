#1 process data, this place we choose the boston data
# and according to the preanalysis we choose four features:'LSTAT','PTRATIO', 'RM', 'TAX'

from sklearn import datasets  # import
import numpy as np
import pandas as pd
boston = datasets.load_boston()  # import boston dataset
print(boston.keys())  # ['data','target','feature_names','DESCR', 'filename']
print(boston.data.shape,boston.target.shape)  # shape of the data (506, 13) (506,)
print(boston.feature_names)  # see features
print(boston.DESCR)  # described
print(boston.filename)

# build input
Y = boston.target.reshape(-1,1)
# use pandas
df = pd.DataFrame(boston.data,columns=boston.feature_names,dtype=float)
column_sels = ['LSTAT','PTRATIO', 'RM', 'TAX']   # choose the feature we want
X1 = df.loc[:,column_sels]
X0 = np.ones(Y.size).reshape(-1,1)  # add constant term
X = np.hstack((X0,X1))
print(X.shape)
print(Y.shape)

# preprocess
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()  # 进行归一化处理
X = min_max_scaler.fit_transform(X)

#2 design the algorithm
# how to use:
# input:X,Y should be array
# each row represents a sample and Y is a column vector
# the input should have constant term which equals to 1
# when the dimension have a huge difference, preprocessing is needed
# last warm warning: you should know your problem first then to choose the model

# cost function
def cost_function(theta,X,Y):
    diff = np.dot(X,theta)-Y   # calculate the difference
    cost = (1/(2*Y.size))*np.dot(diff.transpose(),diff)
    cost = int(cost)
    return cost

# simultaneously calculates the derivative
def gradient_function(theta,X,Y):
    diff = np.dot(X,theta)-Y   # calculate the difference, which is also shown in cost function
    derivative = (1/Y.size)*np.dot(X.transpose(),diff)
    return derivative

# main program: Complete the iteration of gradient descent
def gradient_descent(X,Y):
    import matplotlib.pyplot as plt
    alpha = 0.001
    number,dimension = X.shape
    theta = np.array([1]*dimension).reshape(dimension,1)  # initial value
    gradient = gradient_function(theta,X,Y)  # Using the data and gradients, iterate the optimal coefficients
    number = 1
    costs = []
    numbers =[]
    while not all(abs(gradient)<=1e-5):  # According to the gradient, set the iteration end point
        number += 1
        theta0 = theta
        gradient = gradient_function(theta,X,Y)
        costbefore = cost_function(theta,X,Y)
        theta = theta - alpha*gradient
        costafter = cost_function(theta,X,Y)
        costs.append(costbefore)
        numbers.append(number)
        rate = costafter/costbefore
        if rate > 1: # if the learning rate is too large, then we adjust it automatically
            theta = theta0
            alpha = alpha /2
        if number >40000:  # If more than 40,000 times do not get the optimal result, stop and return the current theta
            import matplotlib.pyplot as plt
            plt.plot(numbers,costs)
            return theta
    plt.plot(numbers,costs)
    print(costafter)
    return theta
    
 # to use the method we write
 import numpy as np
    theta = gradient_descent(X,Y)
    print(theta)