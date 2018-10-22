import numpy as np
import scipy.sparse as sp
np.random.seed(12)
from logistic_regression import Logistic_Regression
from sklearn import linear_model
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

"""
import pickle
def read_t(t,root="./"):
    data = pickle.load(open(root+'Ising2DFM_reSample_L40_T=%.2f.pkl'%t,'rb'))
    return np.unpackbits(data).astype(int).reshape(-1,1600)
"""

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






