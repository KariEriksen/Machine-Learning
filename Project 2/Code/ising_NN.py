import numpy as np
import pickle
import os
import glob
import random
from logistic_regression import Logistic_Regression
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from bp_NN import NeuralNetwork

# define Ising model aprams
# system size
L=40
lambda_ = 0.001
#method = 1 # plot phase of training data
method = 2 # logistic regression with gradient descent on ordered and disordered data
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

del data,labels

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

NN = NeuralNetwork(X_train, Y_train)
NN.train()
