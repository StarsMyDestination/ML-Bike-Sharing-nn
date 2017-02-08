# import math
import numpy as np

# print math.exp(1)


def sigmoid(x):
    '''
    x needs to be a numpy array or list or scale number
    return a numpy array
    '''
    try:
        size = x.shape  # numpy array
        x = x.ravel()
    except AttributeError:
        try:
            size = (len(x), )  # list
        except TypeError:
            size = (1, )
            x = [x]
    tmp = [1 / (1 + np.exp(-_x)) for _x in x]
    return np.array(tmp).reshape(size)


def sigmoid_s(x):
    return 1 / (1 + np.exp(-x))


x = np.arange(0, 10)
print [sigmoid(_x) for _x in x]

ac_fun = sigmoid

print ac_fun(0)

inputs = np.array(x, ndmin=2).T
print inputs.shape

print
print
w = np.random.normal(0, 0.1, (3, 4))
x = np.arange(0, 12).reshape(4, 3)
print w
print x
print np.matmul(w, x)

print
a = np.array([1, 2, 3, 4]).reshape(2, -1)
a = np.array([a])
print a
a = a.ravel()

print
a = [1, 2, 3]
a = np.array([1, 2, 3, 4]).reshape(2, -1)
# print len(a)
print sigmoid(a)
print type(sigmoid(a))
print sigmoid(a).shape
# print a.shape

print
print
a = np.array([1, 3, 4, 4])
print a**2

print
print
a = np.array([1, 3, 4, 4])
print a / 4.0

print
print
# element wise multiply
a = np.arange(0, 6).reshape(2, 3)
b = np.arange(1, 7).reshape(2, 3)
print a
print b
print np.multiply(a, b)
print 1 - a

print
print
a = np.array([1, 2, 3, 4, 5, 6]).reshape(3, -1)
print a
print sigmoid_s(a)

print
print
a = np.array([1, 2])
b = np.array([3, 4])
print a * b
print a.reshape(1, -1).T


print
print
a = np.array([[1, 2, 3], [3, 4, 5]])
print a
b = np.array([1,1,1], ndmin=2).T
print b
c = np.matmul(a, b)
print c
print c.shape
print c.ravel()
