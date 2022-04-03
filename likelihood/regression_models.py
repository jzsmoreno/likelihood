import numpy as np
import matplotlib.pyplot as plt
from likelihood.main import *
from likelihood.tools import *

#-------------------------------------------------------------------------

class fourier_regression:
    """A class that implements the arima model with FFT noise filtering

    Parameters
    ----------
    
    datapoints : np.array
        A set of points to train the arima model.
        
    n_steps : int
        Is the number of points that in predict(n_steps) 
        stage will estimate foward. By default it is set to `0`.
        
    Returns
    -------
    
    new_datapoints : np.array
        It is the number of predicted points. It is necessary 
        to apply predict(n_steps) followed by fit()

    """

    def __init__(self, datapoints, n_steps=0):
        self.datapoints = datapoints
        self.n_steps = n_steps

    def fit(self, sigma = 0, mov = 200, mode = False):
        self.sigma = sigma
        self.mode = mode
        self.mov = mov

        datapoints = self.datapoints
        self.datapoints = fft_denoise(self.datapoints, sigma, mode)

    def predict_save(self, n_steps = 0, n_walkers = 1, mov = 200, name = 'fourier_model'):

        self.n_steps = n_steps
        self.n_walkers = n_walkers
        self.name = name

        new_datapoints = []
        for i in range(self.datapoints.shape[0]):
            model = arima(self.datapoints[i, :])
            model.train(n_walkers, 0, self.mov)
            model.save_model(name)
            y_pred = model.predict(n_steps)
            new_datapoints.append(y_pred)
        
        new_datapoints = np.array(new_datapoints)
        new_datapoints = np.reshape(new_datapoints, (len(new_datapoints), -1))
            
        return new_datapoints

    def load_predict(self, name = 'fourier_model', n_steps = 0, n_walkers = 1):
        self.n_steps = n_steps
        self.n_walkers = n_walkers
        new_datapoints = []
        for i in range(self.datapoints.shape[0]):
            model = arima(self.datapoints[i, :])
            model.load_model(self.name)
            y_pred = model.predict(n_steps)
            new_datapoints.append(y_pred)
        
        new_datapoints = np.array(new_datapoints)
        new_datapoints = np.reshape(new_datapoints, (len(new_datapoints), -1))
            
        return new_datapoints

    def plot_pred(self, y_real, y_pred, ci = 0.65, mode = True):
        plt.figure()
        n = self.n_steps
        y_mean = np.mean(y_pred, axis=0)
        y_std = np.std(y_pred, axis=0)
        if y_mean == 0:
            y_mean = 1
        plt.plot(y_pred, label = 'Predicted')
        plt.plot(y_real, '.--', label = 'Real', alpha = 0.5)
        plt.fill_between((range(y_pred.shape[0]))[-n:]
        , (y_pred - ci*(y_std/y_mean))[-n:]
        , (y_pred + ci*(y_std/y_mean))[-n:], alpha=0.2)
        plt.xlabel('Time steps')
        plt.ylabel('y')
        plt.legend()
        if mode:
            plt.savefig('pred_'+str(n)+'.png', dpi=300)
        plt.show()

class arima: 
    """A class that implements the arima model
    
    Parameters
    ----------
    
    datapoints : np.array
        A set of points to train the arima model.
        
    n_steps : int
        Is the number of points that in predict(n_steps) 
        stage will estimate foward. By default it is set to `0`.
        
    Returns
    -------
    
    y_pred : np.array
        It is the number of predicted points. It is necessary 
        to apply predict(n_steps) followed by train()
    """
    def __init__(self, datapoints, n_steps=0):
        self.datapoints = datapoints
        self.n_steps = n_steps 

    def arima_model(self, datapoints, theta, mode=True, noise=0.0):
        datapoints = self.datapoints
        self.noise = noise
        self.theta_trained = theta

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

    def train(self, nwalkers = 1, noise = 0, mov = 200):

        datapoints = self.datapoints
        xvec = self.xvec
        
        self.nwalkers = nwalkers
        self.noise = noise
        self.mov = mov

        arima_model = self.arima_model

        n = datapoints.shape[0]

        theta = np.ones(shape = n)
        
        x_vec = xvec(datapoints)

        par, error = walkers(nwalkers, x_vec, datapoints, arima_model,
                             theta, mov = mov, figname = None)

        index = np.where(error == np.min(error))[0][0]
        trained = np.array(par[index])

        self.theta_trained = trained

    def predict(self, n_steps = 0):

        self.n_steps = n_steps

        datapoints = self.datapoints
        arima_model = self.arima_model
        theta_trained = self.theta_trained

        y_pred = arima_model(datapoints, theta_trained)

        for i in range(n_steps):
            self.datapoints = y_pred[i:]

            y_new = arima_model(datapoints, theta_trained, mode = False)
            y_pred = y_pred.tolist()
            y_pred.append(y_new)
            y_pred = np.array(y_pred)
            
        return np.array(y_pred)
    
    def save_model(self, name):
        np.savetxt(name+'.txt', self.theta_trained)

    def serialize(self, name):
        np.save(name, self.theta_trained)

    def load_serialized(self, name):
        self.theta_trained = np.load(name)
    
    def load_model(self, name):
        self.theta_trained = np.loadtxt(name+'.txt')

    def eval(self, y_val, y_pred):
        rmse = np.sqrt(np.mean((y_pred - y_val)**2))
        square_error = np.sqrt((y_pred - y_val)**2)
        accuracy = np.sum(square_error[np.where(square_error < rmse)])
        accuracy /= np.sum(square_error)
        print("Accuracy: {:.4f}".format(accuracy))
        print("RMSE: {:.4f}".format(rmse))

    def plot_pred(self, y_real, y_pred, ci = 0.95, mode = True):
        plt.figure()
        n = self.n_steps
        y_mean = np.mean(y_pred, axis=0)
        y_std = np.std(y_pred, axis=0)
        if y_mean == 0:
            y_mean = 1
        plt.plot(y_pred, label = 'Predicted')
        plt.plot(y_real, '.--', label = 'Real', alpha = 0.5)
        plt.fill_between((range(y_pred.shape[0]))[-n:]
        , (y_pred - ci*(y_std/y_mean))[-n:]
        , (y_pred + ci*(y_std/y_mean))[-n:], alpha=0.2)
        plt.xlabel('Time steps')
        plt.ylabel('y')
        plt.legend()
        if mode:
            plt.savefig('pred_'+str(n)+'.png', dpi=300)
        plt.show()