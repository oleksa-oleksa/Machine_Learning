import numpy as np
# 1. Code understanding
print([int(1), float(1), str(1)])

print([-0.5*2**4, str(643)[-2]] + [tuple(2*[3]), (6 != 1) == True])
print([2, 3] + [4, 5])


def pmae(X, Y, p=0.5):
    sm = 0
    for n, x in enumerate(X):
        sm += (x - Y[n])
    sm /= (n + 1)
    return tuple([sm**p, p])


x = np.arange(1, 10)
y = np.random.rand(9)

pmae(x, y)



# numpy.arange([start, ]stop, [step, ]dtype=None)
# Values are generated within the half-open interval [start, stop)
# (in other words, the interval including start but excluding stop).
A = np.arange(-2, 10, 4).reshape((1, 3))
print("A:", A)

B = np.arange(-100, 101, 10).reshape((7, 3))
#print(B)
C = np.arange(100, -101, -10).reshape((7, 3))
#print(C)

D = np.array([1, 2, 3])**2
#print(D)
D = D[:, None]
print("D:", D)
#D = D.reshape((3))
#print(D)
print(0.5*A + D)

D2 = dict([(0, 1), ('1', 2), ('2', 6), (3, 6), (.1, 10)])
print(D2)
res = 0
for k in D2:
    if type(k) == str:
        res += int(k)
        #print(int(k))
    elif type(k) == int:
        res += D2[k]
        #print(D2[k])
    else:
        res /= len(D2)
        print("else")
        #print(len(D2))
    print("k:", k)
    print("res:", res)


class Bernoulli():
    def fit(self, X):
        self.D = {1: 'Yes', 0: 'No'}
        conv = lambda x: 1 if x == 'H' else 0
        res = [conv(x) for x in X]
        self.p = sum(res) / len(res)
    def info(self):
        dec = False
        if self.p < 0.6 and self.p > 0.4:
            dec = True
        return f"The coin is fair: {self.D[int(dec)]} with p value equal {self.p:.3f}"


B = Bernoulli()
B.fit(['H', 'H', 'H', 'T', 'H', 'T', 'T', 'H', 'T', 'T'])
print(B.info())

# ======= Pure Python => Numpy


def slow1(X):
    n, m = X.shape
    D = np.empty_like(X)
    D[0] = [x**2 for x in X[1]]
    for i in range(m):
        D[1, i] = abs(D[0, i] - X[1, i])
    return D


def fast1(X):
    D = np.empty_like(X)
    D[0] = X[1]**2
    D[1] = np.abs(D[0] - X[1])
    return D


X = np.arange(10).reshape(2, 5)
print(slow1(X))
print(fast1(X))


def slow2(X):
    N, d = X.shape
    for j in range(d):
        s = 0
        for i in range(N):
            s += X[i, j]
        for i in range(N):
            X[i, j] -= s/N
    return X


def fast2(X):
    N, d = X.shape
    return X - np.mean(X)


print(np.sum(slow2(X) - fast2(X)))

from math import log


def slow3(X, C):
    N, f = X.shape
    M, f = C.shape
    D = np.zeros((N, M))
    for n in range(N):
        for m in range(M):
            for k in range(f):
                a, b = X[n, k], C[m, k]
                D[n,m] += abs(a-b)
    for n in range(N):
        for m in range(M):
            D[n, m] = log(1 + D[n, m])
    return D


def fast3(X, C):
    D = np.zeros((X.shape[0], C.shape[0]))
    D = np.abs(X - C)
    print("D shape:", D.shape)
    return np.log(1 + D)


X2 = np.arange(12).reshape(3, 4)
C2 = np.arange(12).reshape(3, 4)
#print(np.sum(slow3(X2, C2) - fast3(X2, C2)))

#print(slow3(X2, C2))
#print(fast3(X2, C2))

# =========
T = {'A': 'BCD', 'B': 'AC', 'C': 'AB', 'D': 'A'}
names = {'A': 'lobby', 'B': 'kitchen', 'C': 'bedroom', 'D': 'bathroom'}
import random
initial_state = 'A' # starting always with A
visited_states = [initial_state]

for x in range(1, 1000): # initial_state included
    next_state = random.choice(T[initial_state])
    visited_states += next_state
    initial_state = next_state

print(visited_states[:10])


#===========
n_states = dict()
for key in T:
    n_states[key] = visited_states.count(key)
key_min = min(n_states.keys(), key=(lambda k: n_states[k]))
print('State:', names[key_min], 'visited times:', n_states[key_min])