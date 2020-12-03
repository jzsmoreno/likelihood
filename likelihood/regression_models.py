import numpy as np
from likelihood.main import *

#-------------------------------------------------------------------------


class arima: 
    """A class that implements the arima model
    
    Parameters
    ----------
    
    datapoints : np.array
        A set of points to train the arima model.
        
    n_steps : int
        Is the number of points that in predict(n_steps) 
        stage will estimate foward. By default it is set to `0`.
        
    nwalkers : int
        The number of walkers to be executed.
        
    noise : float
        The amount of noise to add. By default it is set to `0.0`.
        
    Returns
    -------
    
    y_pred : np.array
        It is the number of predicted points. It is necessary 
        to apply predict(n_steps) followed by train()
    """
    def __init__(self, datapoints, n_steps=0, theta_trained=0, 
                 nwalkers=100, noise=0.0):
        self.datapoints = datapoints
        self.n_steps = n_steps 

    def arima_model(self, datapoints, theta, mode=True, noise=0.0):
        datapoints = self.datapoints
        noise = self.noise

        if mode:
            y_vec = []

            y_sum = datapoints 
            y_t = np.dot(theta, y_sum)

            n = y_sum.shape[0]

            for i in range(n):
                try:
                    n_int = np.where(y_sum != y_sum[i])[0]
                    y_i = (y_t - np.dot(theta[n_int], y_sum[n_int]))/theta[i]
                    y_i += np.random.rand()*noise
                except:
                    y_i = (y_t - np.dot(theta[0:i], y_sum[0:i]))/theta[i]
                y_vec.append(y_i)
        else:
            y_sum = datapoints
            y_t = np.dot(theta, y_sum) + y_sum[0]
            n_int = np.where(y_sum != y_sum[0])[0]
            y_i = (y_t - np.dot(theta[n_int], y_sum[n_int]))/theta[0]
            y_i += np.random.rand()*noise
            return y_i

        return np.array(y_vec)

    def xvec(self, datapoints, n_steps = 0):
        datapoints = self.datapoints
        self.n_steps = n_steps

        return datapoints[n_steps:]

    def train(self, nwalkers = 1, noise = 0):

        datapoints = self.datapoints
        xvec = self.xvec
        
        self.nwalkers = nwalkers
        self.noise = noise

        regression_model = self.regression_model

        n = datapoints.shape[0]

        theta = np.ones(shape = n)
        
        x_vec = xvec(datapoints)

        par, error = walkers(nwalkers, x_vec, datapoints, regression_model,
                             theta, mov = 200, figname = None)

        index = np.where(error == np.min(error))[0][0]
        trained = np.array(par[index])

        self.theta_trained = trained

    def predict(self, n_steps = 0):

        self.n_steps = n_steps

        datapoints = self.datapoints
        xvec = self.xvec
        regression_model = self.regression_model
        theta_trained = self.theta_trained

        y_pred = regression_model(datapoints, theta_trained)

        for i in range(n_steps):
            self.datapoints = y_pred[i:]

            y_new = regression_model(datapoints, theta_trained, mode = False)
            #y_new += np.mean(datapoints)
            y_pred = y_pred.tolist()
            y_pred.append(y_new)
            y_pred = np.array(y_pred)
            
        return np.array(y_pred)