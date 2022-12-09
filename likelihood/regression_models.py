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
        self.datapoints_ = datapoints
        self.n_steps = n_steps

    def fit(self, sigma = 0, mov = 200, mode = False):
        self.sigma = sigma
        self.mode = mode
        self.mov = mov

        datapoints = self.datapoints_
        self.datapoints_ = fft_denoise(datapoints, sigma, mode)

    def predict(self, n_steps = 0, n_walkers = 1, name = 'fourier_model', save = True):

        self.n_steps = n_steps
        self.n_walkers = n_walkers
        self.name = name
        mov = self.mov

        assert self.n_walkers <= mov, 'n_walkers must be less or equal than mov'

        new_datapoints = []
        for i in range(self.datapoints_.shape[0]):
            model_ = regressive_arima(self.datapoints_[i, :])
            model_.train(n_walkers, mov)
            if save:
                model_.save_model(str(i)+'_'+name)
            y_pred_ = model_.predict(n_steps)
            new_datapoints.append(y_pred_)
        
        new_datapoints = np.array(new_datapoints)
        new_datapoints = np.reshape(new_datapoints, (len(new_datapoints), -1))
            
        return new_datapoints

    def load_predict(self, name = 'fourier_model'):
        n_steps = self.n_steps

        new_datapoints = []
    
        for i in range(self.datapoints_.shape[0]):
            model_ = regressive_arima(self.datapoints_[i, :])
            model_.load_model(str(i)+'_'+name)
            y_pred_ = model_.predict(n_steps)
            new_datapoints.append(y_pred_)
        
        new_datapoints = np.array(new_datapoints)
        new_datapoints = np.reshape(new_datapoints, (len(new_datapoints), -1))
            
        return new_datapoints

    def plot_pred(self, y_real, y_pred, ci = 0.90, mode = True):
        plt.figure()
        n = self.n_steps
        y_mean = np.mean(y_pred, axis=0)
        y_std = np.std(y_pred, axis=0)
        ci = ci - 0.68
        if ci < 0.95:
            Z = (ci/0.90)*1.64
        else:
            Z = (ci/0.95)*1.96

        plt.plot(y_pred, label = 'Predicted')
        plt.plot(y_real, '.--', label = 'Real', alpha = 0.5)
        plt.fill_between((range(y_pred.shape[0]))[-n:]
        , (y_pred - Z*y_std)[-n:]
        , (y_pred + Z*y_std)[-n:], alpha=0.2)
        plt.xlabel('Time steps')
        plt.ylabel('y')
        plt.legend()
        print('Confidence Interval: {:.4f}'.format(Z*y_std))
        if mode:
            plt.savefig('pred_'+str(n)+'.png', dpi=300)
        plt.show()

class arima:
    """A class that implements the (p, d, q) arima model
    
    Parameters
    ----------
    
    datapoints : np.array
        A set of points to train the arima model.
        
    p : int
        Is the number of auto-regressive terms (ratio). By default it is set to `1`

    d : int
        Is known as the degree of differencing. By default it is set to `0`

    q : int
        Is the number of forecast errors in the model (ratio). By default it is set to `1`
        
    Returns
    -------
    
    y_pred : np.array
        It is the number of predicted points. It is necessary 
        to apply predict(n_steps) followed by train()
    """
    def __init__(self, datapoints, p=1, d=0, q=1, n_steps=0, noise=0, alpha = 0.1):
        self.datapoints = datapoints
        self.n_steps = n_steps
        self.noise = noise
        self.alpha = alpha
        self.p = int(p*len(datapoints))
        self.d = d
        self.q = int(q*len(datapoints))

    def forward(self, y_sum, theta, mode, noise):
        if mode:
            y_vec = []

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
            y_t = np.dot(theta, y_sum) + y_sum[0]
            n_int = np.where(y_sum != y_sum[0])[0]
            y_i = (y_t - np.dot(theta[n_int], y_sum[n_int]))/theta[0]
            y_i += np.random.rand()*noise
            return y_i

        return np.array(y_vec)

    def integrated(self, datapoints):
        datapoints = self.datapoints
        n = datapoints.shape[0]

        y_sum = [((1.0-datapoints[i-1]/datapoints[i])**self.d)*datapoints[i] for i in range(1, n)]
        y_sum.insert(0, datapoints[0])

        return np.array(y_sum)

    def average(self, datapoints):
        y_sum_average = cal_average(datapoints, self.alpha)
        y_sum_eps = datapoints-y_sum_average

        return y_sum_eps

    def abstract_model(self, datapoints, theta, mode=True):
        datapoints = self.datapoints
        noise = self.noise
        self.theta_trained = theta

        phi = theta[0:self.p]
        if self.d != 0:
            y_sum = self.integrated(datapoints)
        else:
            y_sum = datapoints

        y_sum_regr = y_sum[-self.p:]
        y_regr_vec = self.forward(y_sum_regr, phi, mode, noise)
        if self.q != 0:
            y_sum_average = self.average(y_sum[-self.q:])
            y_average_vec = self.forward(y_sum_average, theta[-self.q:], mode, noise)
            check = 0
        else:
            check = None
        
        if check == 0:
            if mode:
                y_vec = y_regr_vec
                for i in reversed(range(y_average_vec.shape[0])):
                    y_vec[i] += y_average_vec[i]
            else:
                y_vec = y_regr_vec+y_average_vec
        else:
            y_vec = y_regr_vec
        
        return y_vec


    def xvec(self, datapoints, n_steps = 0):
        datapoints = self.datapoints
        self.n_steps = n_steps

        return datapoints[n_steps:]

    def train(self, nwalkers = 1, mov = 200, weights = False):

        datapoints = self.datapoints
        xvec = self.xvec
        
        self.nwalkers = nwalkers
        self.mov = mov

        assert self.nwalkers <= self.mov, 'n_walkers must be less or equal than mov' 

        arima_model = self.abstract_model

        n = self.p+self.q
        
        if n == 0:
            n = datapoints.shape[0]

        theta = np.random.rand(n)
 
        x_vec = xvec(datapoints)

        if weights: 
            par, error = walkers(nwalkers, x_vec, datapoints, arima_model,
                             theta = self.theta_trained, mov = mov, tol = 1e-4, figname = None)
        else:
            par, error = walkers(nwalkers, x_vec, datapoints, arima_model,
                             theta, mov = mov, tol = 1e-4, figname = None)

        index = np.where(error == np.min(error))[0][0]
        trained = np.array(par[index])

        self.theta_trained = trained

    def predict(self, n_steps = 0):

        self.n_steps = n_steps

        datapoints = self.datapoints
        arima_model = self.abstract_model
        theta_trained = self.theta_trained
        
        y_pred = arima_model(datapoints, theta_trained)
        
        for i in range(n_steps):
            self.datapoints = y_pred[i:]
            y_new = arima_model(datapoints, theta_trained, mode = False)
            y_pred = y_pred.tolist()
            y_pred.append(y_new)
            y_pred = np.array(y_pred)
        
        return y_pred
    
    def save_model(self, name = 'model'):
        np.savetxt(name+'.txt', self.theta_trained)
    
    def load_model(self, name = 'model'):
        self.theta_trained = np.loadtxt(name+'.txt')

    def eval(self, y_val, y_pred):
        rmse = np.sqrt(np.mean((y_pred - y_val)**2))
        square_error = np.sqrt((y_pred - y_val)**2)
        accuracy = np.sum(square_error[np.where(square_error < rmse)])
        accuracy /= np.sum(square_error)
        print("Accuracy: {:.4f}".format(accuracy))
        print("RMSE: {:.4f}".format(rmse))

    def plot_pred(self, y_real, y_pred, ci = 0.90, mode = True):
        plt.figure()
        n = self.n_steps
        y_mean = np.mean(y_pred, axis=0)
        y_std = np.std(y_pred, axis=0)
        if ci < 0.95:
            Z = (ci/0.90)*1.64
        else:
            Z = (ci/0.95)*1.96
    
        plt.plot(y_pred, label = 'Predicted')
        plt.plot(y_real, '.--', label = 'Real', alpha = 0.5)
        plt.fill_between((range(y_pred.shape[0]))[-n:]
        , (y_pred - Z*y_std)[-n:]
        , (y_pred + Z*y_std)[-n:], alpha=0.2)
        plt.xlabel('Time steps')
        plt.ylabel('y')
        plt.legend()
        print('Confidence Interval: {:.4f}'.format(Z*y_std))
        if mode:
            plt.savefig('pred_'+str(n)+'.png', dpi=300)
        plt.show()

    def summary(self):
        print('\nSummary:')
        print('--------')
        print('\nLenght of theta: {}'.format(len(self.theta_trained)))
        print('\nMean of theta: {:.4f}'.format(np.mean(self.theta_trained)))
        print('------------------------------------------------------------------')


class regressive_arima: 
    """A class that implements the auto-regressive arima (1, 0, 0) model
    
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
    def __init__(self, datapoints, n_steps=0, noise=0):
        self.datapoints = datapoints
        self.n_steps = n_steps
        self.noise = noise

    def arima_model(self, datapoints, theta, mode=True):
        datapoints = self.datapoints
        noise = self.noise
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

    def train(self, nwalkers = 1, mov = 200, weights = False):

        datapoints = self.datapoints
        xvec = self.xvec
        
        self.nwalkers = nwalkers
        self.mov = mov

        assert self.nwalkers <= self.mov, 'n_walkers must be less or equal than mov' 

        arima_model = self.arima_model

        n = datapoints.shape[0]

        theta = np.random.rand(n)
        
        x_vec = xvec(datapoints)
        
        if weights: 
            par, error = walkers(nwalkers, x_vec, datapoints, arima_model,
                             theta = self.theta_trained, mov = mov, tol = 1e-4, figname = None)
        else:
            par, error = walkers(nwalkers, x_vec, datapoints, arima_model,
                             theta, mov = mov, tol = 1e-4, figname = None)

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
    
    def save_model(self, name = 'model'):
        np.savetxt(name+'.txt', self.theta_trained)
    
    def load_model(self, name = 'model'):
        self.theta_trained = np.loadtxt(name+'.txt')

    def eval(self, y_val, y_pred):
        rmse = np.sqrt(np.mean((y_pred - y_val)**2))
        square_error = np.sqrt((y_pred - y_val)**2)
        accuracy = np.sum(square_error[np.where(square_error < rmse)])
        accuracy /= np.sum(square_error)
        print("Accuracy: {:.4f}".format(accuracy))
        print("RMSE: {:.4f}".format(rmse))

    def plot_pred(self, y_real, y_pred, ci = 0.90, mode = True):
        plt.figure()
        n = self.n_steps
        y_mean = np.mean(y_pred, axis=0)
        y_std = np.std(y_pred, axis=0)
        if ci < 0.95:
            Z = (ci/0.90)*1.64
        else:
            Z = (ci/0.95)*1.96
    
        plt.plot(y_pred, label = 'Predicted')
        plt.plot(y_real, '.--', label = 'Real', alpha = 0.5)
        plt.fill_between((range(y_pred.shape[0]))[-n:]
        , (y_pred - Z*y_std)[-n:]
        , (y_pred + Z*y_std)[-n:], alpha=0.2)
        plt.xlabel('Time steps')
        plt.ylabel('y')
        plt.legend()
        print('Confidence Interval: {:.4f}'.format(Z*y_std))
        if mode:
            plt.savefig('pred_'+str(n)+'.png', dpi=300)
        plt.show()

    def summary(self):
        print('\nSummary:')
        print('--------')
        print('\nLenght of theta: {}'.format(len(self.theta_trained)))
        print('\nMean of theta: {:.4f}'.format(np.mean(self.theta_trained)))
        print('------------------------------------------------------------------')