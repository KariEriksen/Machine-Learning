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
        data is an array containing all data from experiment
        B = number of bootstraps
        split = amount split to training data in decimal percentage
        
        ######  Method   #######
	
        Resampling method with replacement, takes data sample
        returns the standard deviation and mean of the resampled
        Samples values with replacement and calculates the mean 
        Repeated n times
        Generates sampling distribution of mean
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

        # Do linear regression
        t = len(self.z_test)
        # Ordinary Least Square method
        if self.method == 'OLS':
            m = np.zeros((self.B,t))
            doubleR = np.zeros(self.B)	
            #mean_z =  np.mean(self.z_test)
            for i in range(self.B):
                index = randint(0, C, C)
                X_resample = self.X_training[index]
                z_resample = self.z_training[index]
                lr = My_Linear_Regression(X_resample, self.X_test, z_resample, self.lambda_)
                lr.My_OLS()
                z_predict = lr.My_Predict(self.X_test, False)
                m[i,:] = z_predict

                #doubleR[i] = 1.0 - (sum((self.z_test - z_predict)**2)/sum((self.z_test - mean_z)**2)) 
                #MeanSquaredError = np.mean((self.z_test - z_predict)**2)

            # Calculate different statistical properties
            MSE = np.mean(np.mean((self.z_test - m)**2, axis=1, keepdims=True) )
            bias = np.mean((self.z_test - np.mean(m, axis=1, keepdims=True))**2 )
            variance = np.mean(np.var(m, axis=1, keepdims=True) )
            #R2 = np.mean(doubleR)
            a = (self.z_test - m)**2
            b = (self.z_test - np.mean(self.z_test))**2
            sum1 = a.sum(axis=1)
            sum2 = sum1.sum()
            sum3 = b.sum()
            doubleR = 1.0 - sum2/sum3
            #doubleR = 1.0 - (sum(sum((self.z_test - m))**2))/(sum((self.z_test - np.mean(self.z_test))**2))
            
            """
            mean_z =  1.0/t*sum(mean_z_vector)
            bias =    1.0/t*sum((self.z_test - mean_z)**2)
            var =     1.0/t*sum((mean_z_vector - mean_z)**2)
            MSE =     1.0/t*sum((self.z_test - mean_z_vector)**2)
            doubleR = 1.0 - (sum((self.z_test - mean_z_vector)**2)/sum((self.z_test - mean_z)**2)) 
            """
            print ('Statistical properties')
            print ('                      ') 
            print ('Bias = %s' % bias)
            print ('Variance = %s' % variance) 
            #print ('Error = %s' % error) 
            print ('MSE = %s' % MSE) 
            print ('Bias + Variance = %s' % (bias + variance)) 
            print ('R2 = %s' % doubleR) 

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
        # Calculate the mean and standard deviation
        #mean = 1.0/n*(sum(t))  
        #std = 1.0/n*(sum())
        
        # Calculate the Mean Square Error
        #diff = self.z - self.z_predict
        #diff = np.mean(self.z) - np.mean(m)
        #MSE = 1.0/(n*n)*(diff*diff)
        
        return m, self.z_test




