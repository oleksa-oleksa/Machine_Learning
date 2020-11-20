import numpy,scipy,scipy.io
import matplotlib.pyplot as plt

# ----------------------------------------
# Create a matrix of population density 
# ----------------------------------------

data = scipy.io.loadmat('data.mat')

population = data['population']
countries  = data['countries']

# ----------------------------------------
# Size of the map
# ----------------------------------------
assert(population.shape[0]==countries.shape[0])
assert(population.shape[1]==countries.shape[1])
nx = population.shape[0]
ny = population.shape[1]

# ----------------------------------------
# Plots geographical locations on a map
#
# input:
# - an array of latitudes
# - an array of longitudes
#
# ----------------------------------------
def plot(latitudes,longitudes):
    plt.figure(figsize=(14,10))
    plt.imshow(population**.25,cmap='Blues')
    for a in set(list(countries.flatten())): plt.contour((countries==a)*1.0,colors='black',levels=[0.5])
    plt.plot(longitudes,latitudes,'o',ms=7,markeredgewidth=1,mfc='yellow',mec='black')
    plt.axis([0,population.shape[1],population.shape[0],0])

