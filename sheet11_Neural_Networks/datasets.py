import numpy

def toy():
    """
    Simple One-Dimensional Dataset
    """

    # number of examples
    n = 40

    # function parameters
    a      = 1
    b      = 2.65
    offset = 1
    noise  = 0.2

    rstate = numpy.random.mtrand.RandomState(12345)

    X = rstate.uniform(-3,3,[n,1])

    Y1 = numpy.sin(a*X)+numpy.cos(b*X+offset)
    Y2 = 0.5*numpy.cos(2*b*X-offset)+0.3*numpy.sin(0.1*a*X-2)
    Y3 = rstate.normal(0,noise,X.shape)

    Y = (Y1+Y2+Y3).flatten()

    return X,Y


def yacht():
    """
    Yacht Hydrodynamics Data Set
    """

    D = numpy.loadtxt(open("yacht_hydrodynamics.csv","rb"),delimiter=",",skiprows=1)

    X,Y = D[:,:-1],D[:,-1]

    return X,Y

