import numpy as np
import scipy.sparse as sp
np.random.seed(12)
from linear_regression import My_Linear_Regression
from bootstrap import Bootstrap
from sklearn import linear_model
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

leastsq=linear_model.LinearRegression()

import warnings
#Comment this to turn on warnings
warnings.filterwarnings('ignore')

# define Ising model aprams
# system size
L=40
lambda_ = 1.0

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
# calculate Ising energies
energies=ising_energies(states,L)

# reshape Ising states into RL samples: S_iS_j --> X_p
states=np.einsum('...i,...j->...ij', states, states)
shape=states.shape
states=states.reshape((shape[0],shape[1]*shape[2]))
# build final data set

Data=[states,energies]
coefs_leastsq = []

# define number of samples
n_samples=400
# define train and test data sets
X_train=Data[0][:n_samples]
Y_train=Data[1][:n_samples] #+ np.random.normal(0,4.0,size=X_train.shape[0])
X_test=Data[0][n_samples:3*n_samples//2]
Y_test=Data[1][n_samples:3*n_samples//2] #+ np.random.normal(0,4.0,size=X_test.shape[0])

lr = My_Linear_Regression(X_train, X_test, Y_train, lambda_)
lr.My_OLS()
energies_predict = lr.My_Predict(Y_test, False)
coeff = lr.My_Beta()
"""
### ordinary least squares
leastsq.fit(X_train, Y_train) # fit model 
coefs_leastsq.append(leastsq.coef_) # store weights
"""
J_leastsq=np.array(coeff).reshape((L,L))
    
cmap_args=dict(vmin=-1., vmax=1., cmap='seismic')
#fig, axarr = plt.subplots(nrows=1, ncols=3)
plt.imshow(J_leastsq,**cmap_args)
#axarr[0].set_title('$\\mathrm{OLS}$',fontsize=16)
#axarr[0].tick_params(labelsize=16)

plt.show()






