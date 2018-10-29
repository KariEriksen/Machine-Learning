import numpy as np
import pickle
import os
import glob
import random
from logistic_regression import Logistic_Regression
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# define Ising model aprams
# system size
L=40
lambda_ = 0.001
#method = 1 # plot phase of training data
method = 2 # logistic regression with gradient descent
#method = 3 # logistic regression with stochastic gradient descent

# path to data directory
cwd = os.getcwd()
path_to_data=cwd + '/IsingData/'

# load data
file_name = "Ising2DFM_reSample_L40_T=All.pkl" # this file contains 16*10000 samples taken in T=np.arange(0.25,4.0001,0.25)

data = pickle.load(open(path_to_data+file_name,'rb')) # pickle reads the file and returns the Python object (1D array, compressed bits)
data = np.unpackbits(data).reshape(-1, 1600) # Decompress array and reshape for convenience
data=data.astype('int')
data[np.where(data==0)]=-1 # map 0 state to -1 (Ising variable can take values +/-1)

file_name = "Ising2DFM_reSample_L40_T=All_labels.pkl" # this file contains 16*10000 samples taken in T=np.arange(0.25,4.0001,0.25)
labels = pickle.load(open(path_to_data+file_name,'rb')) # pickle reads the file and returns the Python object (here just a 1D array with the binary labels)

# divide data into ordered, critical and disordered
X_ordered=data[:70000,:]
Y_ordered=labels[:70000]

X_critical=data[70000:100000,:]
Y_critical=labels[70000:100000]

X_disordered=data[100000:,:]
Y_disordered=labels[100000:]

# define training and test data sets
X=np.concatenate((X_ordered,X_disordered))
Y=np.concatenate((Y_ordered,Y_disordered))

rand = np.arange(len(Y))
np.random.shuffle(rand)

X = X[rand]
Y = Y[rand]

X_train = X[:9000,:]
Y_train = Y[:9000]
X_test = X[9000:,:]
Y_test = Y[9000:]
beta = np.random.normal(0, 1, np.size(X_train,1))

if method == 1:
	# plot
	cmap_args=dict(vmin=-1., vmax=1., cmap='seismic')
	plt.subplot(311)
	plt.imshow(X_ordered[20000].reshape(L,L),**cmap_args)
	plt.title('Ordered',fontsize=12)
	plt.tick_params(labelsize=10)
	
	plt.subplot(312)
	plt.imshow(X_critical[10000].reshape(L,L),**cmap_args)
	plt.title('Critical',fontsize=12)
	plt.tick_params(labelsize=10)

	plt.subplot(313)
	plt.imshow(X_disordered[50000].reshape(L,L),**cmap_args)
	plt.title('Disordered',fontsize=12)	
	plt.tick_params(labelsize=10)

	#plt.subplots_adjust(wspace=0.05, hspace=0.5)
	plt.show()

if method == 2:
	# Gradient descent 
	eps = 1e-5
	e = 1.0
	eta = 0.5
	n = 100
	log_r = Logistic_Regression(X_train, X_test, Y_train, lambda_)
	for i in range(n):
		if e > eps:
			# Calculate the derivative of the cost function	
			gradient = log_r.deri_cross_entropy(beta)
			v_t = eta*gradient
			#print (gradient)
			beta = beta - v_t
			e = abs(np.mean(gradient))
			#print (e)
	indicator = 0.0
	for j in range(n):
		Y_pred = beta[j]*X_test[j]	
		if Y_pred == Y_train[j]:
			indicator += 1.0
	accuracy = indicator/n
	#print (beta)
	
if method == 3:
	# Stochastic gradient descent 
	eps = 1e-5
	e = 1.0
	eta = 0.5
	epochs = 100
	M = 200
	m = int(n/M)
	for epoch in range(epochs):
		for i in range(m):
			if e > eps:
				k = np.random.randint(m)
				log_r = Logistic_Regression(X_train[k:k+1,:], X_test[k:k+1,:], Y_train[k:k+1,:], lambda_)
				# Calculate the derivative of the cost function	
				gradient = log_r.deri_cross_entropy(beta)
				v_t = eta*gradient
				beta = beta - v_t
				e = abs(np.mean(gradient))
	
	




