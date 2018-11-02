import numpy as np
from numpy.random import randint

"""
m = np.zeros((3,10))
for i in range(3):
	m[i,:] = i+1
	print (m[i,:])

Y_test = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
Y_mat = np.tile(Y_test, (3, 1))

print Y_mat

print Y_mat - m

print np.mean(Y_mat)

print np.mean(Y_mat, axis=1)

print Y_mat.mean()
"""

a = randint(0, 10, 10)

W = 2*np.random.random(10) - 1 
print (randint(0, 5, 5))
