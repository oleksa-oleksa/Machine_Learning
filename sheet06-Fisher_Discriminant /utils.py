import numpy,matplotlib,matplotlib.pyplot

class Abalone:

	# Instantiate the Abalone dataset
	#
	# input: None
	# output: None
	#
	def __init__(self):
		X = numpy.genfromtxt('abalone.csv',delimiter=',')
		self.M = X[X[:,0]==0][:,1:-1]
		self.F = X[X[:,0]==1][:,1:-1]
		self.I = X[X[:,0]==2][:,1:-1]

	# Plot a histogram of the projected data
	#
	# input: the projection vector and the name of the projection
	# output: None
	#
	def plot(self,w,name):
	    matplotlib.pyplot.xlabel(name)
	    matplotlib.pyplot.ylabel('# examples')
	    matplotlib.pyplot.hist(numpy.dot(self.M,w),histtype="stepfilled",bins=25, alpha=0.8, normed=True,color='blue')
	    matplotlib.pyplot.hist(numpy.dot(self.F,w),histtype="stepfilled",bins=25, alpha=0.8, normed=True,color='red')
	    matplotlib.pyplot.hist(numpy.dot(self.I,w),histtype="stepfilled",bins=25, alpha=0.8, normed=True,color='green')
	    matplotlib.pyplot.legend(['male','female','infant'])
	    matplotlib.pyplot.show()

