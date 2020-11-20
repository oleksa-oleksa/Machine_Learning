import numpy,matplotlib
import matplotlib.pyplot as plt

# Load the handwritten characters dataset
#
# input:  None
# output: a matrix of size nbsamples*nbdims containing the data
#
def load():
    f = open('characters.csv','r')
    X = numpy.array([[int(s) for s in l.split(',')] for l in f],dtype='float32')
    return X

# Render the data as 28x28 images
#
# input:  an array of size n*784
# output: None
#
def render(x):
    assert(x.ndim==2 and x.shape[0] <= 25 and x.shape[1] == 784)
    x = x.reshape([-1,28,28])
    z = numpy.ones([len(x),30,30])*x.max()
    z[:,1:-1,1:-1] = x
    n = len(x)**.5; assert(n%1==0); n = int(n)
    x = z.reshape([n,n,30,30]).transpose([0,2,1,3]).reshape([n*30,n*30])
    plt.figure(figsize=(5,5))
    plt.imshow(x,cmap=plt.cm.gray)
    plt.axis('off')
    plt.show()

# Scatter plot
#
# input: the vector of x- and y-coordinates and the axes labels
# output: None
#
def scatterplot(x,y,xlabel='',ylabel=''):
    assert(x.ndim==1 and y.ndim==1 and len(x)==len(y))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(x,y,'.')
    plt.show()

