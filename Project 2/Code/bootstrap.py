import numpy as np
from numpy.random import randint
from linear_regression import My_Linear_Regression 

class Bootstrap:
    def __init__(self, X_training, X_test, z, B, lambda_, method):
        self.X_training = X_training
        self.X_test = X_test
        self.z = z
        self.B = B
        self.lambda_ = lambda_	
        self.method = method
	    
    def My_Bootstrap(self):

        """
        Bootstrap class
	
        ###### Variables #######
        X = the design matrix     (n, p) array
        z = the response variable (n, )  array
        B = number of bootstraps
        lambda = the penalty 
        method = the regression method used
        
        ######  Method   #######
	
        Resampling method with replacement, takes data sample
        returns the predicted values of z and the test data z.
        """

        t = len(self.X_test)
	r = np.size(self.X_test,1)
        # Ordinary Least Square method
        if self.method == 'OLS':
            m = np.zeros((self.B,t))
            c = np.zeros((self.B,r))
            for i in range(self.B):
                index = randint(0, t, t)
                X_resample = self.X_training[index]
                z_resample = self.z[index]
                lr = My_Linear_Regression(X_resample, self.X_test, z_resample, self.lambda_)
                lr.My_OLS()
                z_predict = lr.My_Predict(self.X_test, False)
		coeff = lr.My_Beta()
                m[i,:] = z_predict
                c[i,:] = coeff

        # Ridge regression
        elif self.method == 'Ridge':
            m = np.zeros((self.B,t))
            c = np.zeros((self.B,r))	
            for i in range(self.B):
                index = randint(0, t, t)
                X_resample = self.X_training[index]
                z_resample = self.z[index]
                lr = My_Linear_Regression(X_resample, self.X_test, z_resample, self.lambda_)
                lr.My_Ridge()
                z_predict = lr.My_Predict(self.X_test, False)
		coeff = lr.My_Beta()
                m[i,:] = z_predict
                c[i,:] = coeff
        
        #Lasso regression
        elif self.method == 'Lasso':
            m = np.zeros((self.B,t))
            c = np.zeros((self.B,r))	
            for i in range(self.B):
                index = randint(0, t, t)
                X_resample = self.X_training[index]
                z_resample = self.z[index]
                lr = My_Linear_Regression(X_resample, self.X_test, z_resample, self.lambda_)
                lr.My_Lasso()
                z_predict = lr.My_Predict(self.X_test, True)
		coeff = lr.My_Beta()
                m[i,:] = z_predict
                c[i,:] = coeff
        
        else:
            print('You have forgotten to select method; OLS, Ridge or Lasso.')

        return m, c




