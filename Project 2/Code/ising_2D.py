import numpy as np
import pickle
import os
import glob
import scipy.sparse as sp
np.random.seed(12)
from sklearn import linear_model
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# path to data directory
cwd = os.getcwd()
path_to_data=cwd + '/IsingData/'

print (pickle.HIGHEST_PROTOCOL)
# load data
file_name = "Ising2DFM_reSample_L40_T=All.pkl" # this file contains 16*10000 samples taken in T=np.arange(0.25,4.0001,0.25)

data = pickle.load(open(path_to_data+file_name,'rb')) # pickle reads the file and returns the Python object (1D array, compressed bits)
data = np.unpackbits(data).reshape(-1, 1600) # Decompress array and reshape for convenience
data=data.astype('int')
data[np.where(data==0)]=-1 # map 0 state to -1 (Ising variable can take values +/-1)

file_name = "Ising2DFM_reSample_L40_T=All_labels.pkl" # this file contains 16*10000 samples taken in T=np.arange(0.25,4.0001,0.25)
labels = pickle.load(open(path_to_data+file_name,'rb')) # pickle reads the file and returns the Python object (here just a 1D array with the binary labels)

data[data == 0] = -1

ordered = slice(0, 70000)
critical = slice(70000, 100000)
disordered = slice(100000, 160000)

# define Ising model aprams
# system size
L=40
lambda_ = 0.001

# create 10000 random Ising states
states=np.random.choice([-1, 1], size=(10000,L))

def ising_energies(states,L):
    """
    This function calculates the energies of the states in the nn Ising Hamiltonian
    """
    J=np.zeros((L,L),)
    for i in range(L):
	# only nearest neightbors have interaction
        J[i,(i+1)%L]-=1.0
	
    # compute energies, Einstein summation
    E = np.einsum('...i,ij,...j->...',states,J,states)

    return E

energies=ising_energies(states,L)

# reshape Ising states into RL samples: S_iS_j --> X_p
states=np.einsum('...i,...j->...ij', states, states)
shape=states.shape
states=states.reshape((shape[0],shape[1]*shape[2]))
# build final data set

Data=[states,energies]
coefs_leastsq = []
coefs_ridge = []

# define number of samples
n_samples=400
# define train and test data sets
X_train=Data[0][:n_samples]
Y_train=Data[1][:n_samples] #+ np.random.normal(0,4.0,size=X_train.shape[0])
X_test=Data[0][n_samples:3*n_samples//2]
Y_test=Data[1][n_samples:3*n_samples//2] #+ np.random.normal(0,4.0,size=X_test.shape[0])
beta = np.zeros(n_samples)


# Gradient descent 
eps = 1e-5
n = 100
eta = 0.1
log_r = Logistic_Regression(X_train, X_test, Y_train, lambda_)
for i in range(n):
	if e > eps:
		# Calculate the derivative of the cost function
		gradient = log_r.deri_cross_entropy(beta)
		v_t = eta*gradient[i,:]
		beta = beta - v_t
		e = abs(gradient)
plt.show()






