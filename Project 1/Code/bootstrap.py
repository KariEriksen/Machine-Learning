import numpy as np
from numpy.random import randint
from linear_regression import My_Linear_Regression 

class Bootstrap:
    def __init__(self, X, z, B, lambda_, split, method):
        self.X = X
        self.z = z
        self.B = B
        self.lambda_ = lambda_	
        self.split = split
        self.method = method
	    
    def My_Bootstrap(self):

        """
        Bootstrap class
	
        ###### Variables #######
        X = the design matrix     (n, p) array
        z = the response variable (n, )  array
        B = number of bootstraps
        lambda = the penalty 
        split = amount split to training data in decimal percentage
        method = the regression method used
        
        ######  Method   #######
	
        Resampling method with replacement, takes data sample
        returns the predicted values of z and the test data z.
        """

        # Split the data according to the given decimal percentage
        # split = amount of data set to training
        n = np.size(self.X,0)   # size of column (number of rows)
        C = int(n*self.split)

        # Shuffle the datapoint to randomized order
        randomize = np.arange(n)
        np.random.shuffle(randomize)
        self.X = self.X[randomize]
        self.z = self.z[randomize]

        self.X_training = self.X[:C,:]
        self.X_test = self.X[C:,:]
        self.z_training = self.z[:C]
        self.z_test = self.z[C:]

        t = len(self.z_test)
        # Ordinary Least Square method
        if self.method == 'OLS':
            m = np.zeros((self.B,t))
            doubleR = np.zeros(self.B)	
            for i in range(self.B):
                index = randint(0, C, C)
                X_resample = self.X_training[index]
                z_resample = self.z_training[index]
                lr = My_Linear_Regression(X_resample, self.X_test, z_resample, self.lambda_)
                lr.My_OLS()
                z_predict = lr.My_Predict(self.X_test, False)
                m[i,:] = z_predict

        # Ridge regression
        elif self.method == 'Ridge':
            m = np.zeros((self.B,t))	
            for i in range(self.B):
                index = randint(0, C, C)
                X_resample = self.X_training[index]
                z_resample = self.z_training[index]
                lr = My_Linear_Regression(X_resample, self.X_test, z_resample, self.lambda_)
                lr.My_Ridge()
                z_predict = lr.My_Predict(self.X_test, False)
                m[i,:] = z_predict
        
        #Lasso regression
        elif self.method == 'Lasso':
            m = np.zeros((self.B,t))	
            for i in range(self.B):
                index = randint(0, C, C)
                X_resample = self.X_training[index]
                z_resample = self.z_training[index]
                lr = My_Linear_Regression(X_resample, self.X_test, z_resample, self.lambda_)
                lr.My_Lasso()
                z_predict = lr.My_Predict(self.X_test, True)
                m[i,:] = z_predict
        
        else:
            print('You have forgotten to select method; OLS, Ridge or Lasso.')

        return m, self.z_test




