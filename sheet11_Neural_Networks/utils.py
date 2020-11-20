import numpy,scipy.spatial.distance

def gaussianKernel(X1,X2,width):
    """
    Generates the Gaussian kernel matrix K with K[i,j]=k(X1[i,:],X2[j,:])
    """
    return numpy.exp(-scipy.spatial.distance.cdist(X1,X2,metric='sqeuclidean')/(2*width**2))


def split(X,Y):
    """
    Partitions a dataset into a training and test set
    """
    n=len(X)

    rstate = numpy.random.mtrand.RandomState(2345)

    R = rstate.permutation(n)
    Rtrain = R[:n//2]
    Rtest  = R[n//2:]

    Xtrain = X[Rtrain] # Training data
    Ytrain = Y[Rtrain] # Training targets

    Xtest  = X[Rtest] # Test data
    Ytest  = Y[Rtest] # Test targets

    return Xtrain,Ytrain,Xtest,Ytest

