import numpy,scipy,scipy.io
import matplotlib.pyplot as plt

def generateData(lam, p1, p2, N, M):
    data = numpy.zeros((N,M),dtype=numpy.int8)
    probability=0 #
    for i in range(0,N):
        if(numpy.random.rand()<=lam):
            probability=p1 # We are using coin A
        else:
            probability=p2 # We are using coin B
            
        data[i] = (numpy.random.rand(M,1)<=probability)[:,0] #Record the M throws
    return data


def unknownData():
	return scipy.io.loadmat('data.mat')['data']

def plot(data,distribution):
	N,M = data.shape
	f = plt.figure(figsize=(7,5))

	# Compute max value of the histogram
	sdata = data.sum(axis=1)
	hmax = 1.5*numpy.max([(sdata == i).sum() for i in range(M+1)])

	# Plot data histogram
	ax1 = f.add_subplot(111)
	ax1.set_ylim(0,hmax)
	ax1.hist(data.sum(axis=1),bins=numpy.arange(M+2)-0.5,alpha=0.3,color='g')
	ax1.set_xlabel('x')
	ax1.set_xlim(-0.5,M+0.5)

	# Plot probability function
	ax2 = ax1.twinx()
	ax2.set_ylim(0,hmax/N)
	ax2.plot(range(M+1),distribution,'-o',c='r')
	ax2.set_ylabel('p(x)')
	ax2.set_xlim(-0.5,M+0.5)

