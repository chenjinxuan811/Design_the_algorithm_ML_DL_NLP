import numpy as np

class KNN:
    def __init__(self,X,y,k=3,p=2):
        """
        Function:give the index of the new sample use KNN algorithm

        about the design:
        Here's how we handle it: separate data from target to make the data processing clearer
        Here we use classes to implement: because classes can save the data we enter into their own properties
        At present, single point prediction is realized, and multi-point prediction can be carried out in external circulation

        :param X: a matrix, each row is a sample and column represented feature
        :param y: the index of sample, it is required to be a column vector
        :param k: number of the neighbor
        :param p: latitude of norm, to measure the distance,the default is European distance
        """
        num1,col1 = X.shape
        if num1 != y.size:    # check the data
            raise Exception("fail to create the class! dimension of X,y doesn't match!")
        self.data = X
        self.target = y
        self.col = col1
        self.num = num1
        self.k = k
        self.p = p

    def update_k(self,k):  # change the number of the neighbors
        self.k = k

    def get_k(self,k):  # check the number of the neighbors
        return k

    def _distance(self,X):
        """
        private method: to calculate the distance of two samples
        """
        col = X.size
        if col != self.col:
            raise Exception("dimension unmatched! the dimension of the existed samples is:"+self.col+"\n"+"that of the input is:"+col)
        distance = []      # store the distance
        for k in range(self.num):  # traverse a;; samples
            distance.append(np.linalg.norm(X - self.data[k], ord=self.p))
        return distance

    def fit(self,X):
        """
        fit the data and give the index of the category

        :param X: a row vector
        :return: the index of the category
        """
        distance = self._distance(X)
        label = list(zip(distance,self.target))  # 形成序偶
        sequence = sorted(label)    # 排序的时候，默认按照第一个来排列
        neighbor = []   # 放邻居的标签
        for i in range(self.k):
            neighbor.append(sequence[i][1])
        predicts = {}
        for k in neighbor:
            if k not in predicts.keys():
                predicts[k]=1
            else:
                predicts[k]+=1
        predict = max(predicts.items())[0]
        return predict


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_iris

    # Import dataset
    iris=load_iris()  # What is returned here is a dictionary
    # Segmentation dataset into train data and test data
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    knn = KNN(X_train,y_train)
    num,col = X_test.shape
    count = 0
    # Calculation accuracy
    for i in range(num):
        predict = knn.fit(X_test[i])
        predict = int(predict)
        if predict == y_test[i]:
            count = count + 1
    right = count/num
    print(right)
