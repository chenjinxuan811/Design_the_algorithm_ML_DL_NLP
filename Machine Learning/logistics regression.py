import numpy as np
def sigmoid(x):
    return 1.0/(1+np.exp(-x))

def cost_function(X,Y,ws):
    x = np.dot(X,ws)
    left = np.multiply(Y, np.log(sigmoid(x)))
    right = np.multiply(1 - Y, np.log(1 - sigmoid(x)))
    return np.sum(left + right) / -(len(X))

def logistic_gradient(X,Y,alpha = 0.1):
    """
    train the model

    :param X: array, each row is an sample and each column is a feature
    :param Y: array, label with 0 and 1
    :param alpha: initial learnning rate
    :return: array, with the training parameter
    """
    import matplotlib.pyplot as plt
    # Construct factor
    num,col = X.shape
    ws = np.ones((col,1))
    epochs = 10000  # The number of iterations, since we don't know if the problem is a convex problem, we don't take the gradient size as a termination condition
    # start gradient descend
    time = 0
    costs = []
    times = []
    for i in range(epochs):
        time = time + 1
        h = sigmoid(np.dot(X,ws))
        gradient = np.dot(X.T,(h-Y))/num
        ws0 = ws
        costbefore = cost_function(X,Y,ws)
        ws = ws - alpha*gradient
        costafter = cost_function(X,Y,ws)
        if costafter > costbefore:
            ws = ws0   # the learning rate is not appropriate, change the learning rate
            alpha = alpha/2
        else:
            times.append(time)
            costs.append(costbefore)
    plt.plot(times,costs)
    return ws

# construct input
data = np.genfromtxt("LR-testSet.csv", delimiter=",")
x_data = data[:,:-1]
y_data = data[:,-1].reshape(-1,1)
x_data = np.hstack((np.ones((len(x_data),1)),x_data))
ws = logistic_gradient(x_data,y_data)
y_predict = sigmoid(np.dot(x_data,ws))
right = 0
for i in range(len(y_predict)):
    if y_predict[i]>=0.5:
        y_predict[i]=1
    else:
        y_predict[i]=0
    if y_predict[i] == y_data[i]:
        right += 1
print(ws)
print(right/y_data.size)
