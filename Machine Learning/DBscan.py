import numpy as np

def fit(X,eta,num):
    """
    To find the core and it's density direct point

    :param X:
    :param eta: the distance to define if the point can be a density point
    :param num: determine what kind of sample point could be called core
    :return: dic in which stores all the core object and the point which is density direct
    """
    coreindex = {}  # where store all the core object and the point around them
    # calculate each sample point
    for i in range(X.shape[0]):
        dist = [] # to put all the density direct point
        for k in range(X.shape[0]):
            if i != k:
                distance = np.linalg.norm(X[i,:]-X[k,:], ord=2)
                if distance <= eta:
                    dist.append(k)
        if len(dist)>=num:   # Determine if it is a core object
            coreindex[i]=dist
    return coreindex

# input the core to train
def train(coreindex):
    import random
    classindex = 0  # denote the name of each class
    classdict = {}
    index = list(coreindex.keys())  # store the core that haven't be explored
    while len(index)!=0:
         # find a core randomly
        sample = random.randint(0,len(index)-1)
        sample = index[sample]
        classdict[classindex] = coreindex[sample]   # put the density direct point in
        classdict[classindex].append(sample)  # put the core in
        index.remove(sample)
        for i in classdict[classindex]:  # find around the core
            if i in index:
                index.remove(i)
                for j in coreindex[i]:
                    if j in classdict[classindex]: # can't join the same point twice
                        pass
                    else:
                        classdict[classindex].append(j)
        classindex += 1  # find the next class
    return classdict
