class Kmeans:
    """
    input: dot without label
    output: the class of the dot
    function: inout a new dot, to predict the class of the dot
    """
    def __init__(self):
        pass

    def distance(self, Xi):
        import numpy as np
        dist = []
        for k in self.centroid:  # Iterate over all samples in the model
            dist.append(np.linalg.norm(k - Xi, ord=self.p))
        dist0 = list(enumerate(dist))
        near = sorted(dist0, key=lambda x: x[1])[0][0]
        return near

    def predict(self, Xi):
        import numpy as np
        num = Xi.shape[0]
        dist = []
        for i in range(num):
            dist.append([])
            for k in self.centroid:  # Iterate over all samples in the model
                dist[i].append(np.linalg.norm(k - Xi, ord=self.p))
            dist0 = list(enumerate(dist[i]))
            near = sorted(dist0, key=lambda x: x[1])[0][0]
            dist[i] = near
        return dist

    def fit(self, X, k=3, p=1):  # start training
        import random
        import numpy as np
        import copy
        self.k = k
        self.p = p
        self.centroid = []
        self.data = X
        num, col = X.shape
        pick = random.sample(range(0, num), self.k)  # Randomly take k sample points
        # Initialize the classification result record matrix and centroids
        for i in pick:
            self.centroid.append(X[i, :])
        flag = True
        while flag:
            classlist = []
            for i in range(self.k):
                classlist.append([])
            centroid = self.centroid.copy()  # take down the core before
            for i in range(num):
                near = self.distance(self.data[i, :])
                classlist[near].append(self.data[i, :])
            # update the core
            for i in range(self.k):
                self.centroid[i] = np.average(classlist[i], axis=0)
            if (np.array(self.centroid) - np.array(centroid)).all() == 0:  # if the core do not change, stop the iteration
                flag = False
        return self.centroid, classlist
