from typing import TypeVar, List, Iterator, Callable, Tuple
import numpy as np


"""
Data Science from Scratch, Second Edition, by Joel Grus (O'Reilly).Copyright 2019 Joel Grus, 978-1-492-04113-9
"""

T = TypeVar('V')

def minibatches(dataset: List[T],
                batch_size: int,
                shuffle: bool = True) -> Iterator[List[T]]:
    """Generates 'batch_size'-sized minibatches from the dataset"""
    # start indexes 0, batch_size, 2 * batch_size, ...
    batch_starts = [start for start in range(0, len(dataset), batch_size)]
    
    if shuffle: np.random.shuffle(batch_starts)  # shuffle the batches
            
    for start in batch_starts:
        end = start + batch_size
        yield dataset[start:end]
        
Vector = List[float]
        
def difference_quotient(f: Callable[[float], float],
                        x: float,
                        h: float) -> float:
    return(f(x + h) - f(x)) / h

def partial_difference_quotient(f: Callable[[Vector], float],
                                v: Vector,
                                i: int,
                                h: float) -> float:
    """Returns the i-th partial difference quotient of f at v"""
    w = [v_j + (h if j == i else 0)  # add h to just the ith element of v
         for j, v_j in enumerate(v)]
    return (f(w) - f(v)) / h

def estimate_gradient(f: Callable[[Vector], float],
                      v: Vector,
                      h: float = 0.0001):
    return [partial_difference_quotient(f, v, i, h)
            for i in range(len(v))]

#-------------------------------------------------------------------------

def rescale(dataset, n = 1):
    """Perform a standard rescaling of the data
    
    Parameters
    ----------
    dataset : np.array
        An array containing the model data.
    n : int
        Is the degree of the polynomial to subtract 
        the slope. By default it is set to `1`.
        
    Returns
    -------
    data_scaled : np.array
        An array containing the scaled data.
        
    mu : np.array
        An array containing the mean of the 
        original data.
    sigma : np.array
        An array containing the standard 
        deviation of the original data.
    
    """
    
    mu = []
    sigma = []
    fitting = []
    
    try:
        xaxis = range(dataset.shape[1])
    except:
        error_type = 'IndexError'
        msg = 'Trying to access an item at an invalid index.'
        print(f'{error_type}: {msg}')
        return None
    for i in range(dataset.shape[0]):
        if n != None:
            fit = np.polyfit(xaxis, dataset[i, :, 0], n)
            f = np.poly1d(fit)
            poly = f(xaxis)
            fitting.append(f)
        else:
            fitting.append(0.0)
        dataset[i, :, 0] += -poly
        mu.append(np.min(dataset[i, :, 0]))
        if np.std(dataset[i, :, 0]) != 0: 
            sigma.append(np.std(dataset[i, :, 0]))
        else:
            sigma.append(1)
            
        dataset[i, :, 0] = (dataset[i, :, 0] - mu[i]) / sigma[i]
         
    values = [mu, sigma, fitting]
    
    return dataset, values

def scale(dataset, values):
    """Performs the inverse operation to the rescale function
    
    Parameters
    ----------
    dataset : np.array
        An array containing the scaled data.
    values : np.ndarray
        A set of values returned by the rescale function.
    
    """
    
    for i in range(dataset.shape[0]):
        dataset[i, :, 0] = dataset[i, :, 0]*values[1][i]
        dataset[i, :, 0] += values[0][i]
        dataset[i, :, 0] += values[2][i](range(dataset.shape[1]))
    
    return dataset

def generate_series(n, n_steps, incline = True):
    """Function that generates $n$ series of length $n_steps$
    """
    freq1, freq2, offsets1, offsets2 = np.random.rand(4, n, 1)
    
    if incline :
        slope = np.random.rand(n, 1)
    else: 
        slope = 0.0
        offsets2 = 1
        
    time = np.linspace(0, 1, n_steps)
    series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10)) # wave 1
    series += 0.2 * np.sin((time - offsets2) * (freq2 * 20 + 20)) # + wave 2
    series += 0.7 * (np.random.rand(n, n_steps) - 0.5) # + noise
    series += 5 * slope * time + 2 * (offsets2-offsets1) * time ** (1-offsets2)
    series = series
    return series[..., np.newaxis].astype(np.float32)

from likelihood.__main__ import *

#-------------------------------------------------------------------------

"""Regression models
"""

class regression: 
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
    def __init__(self, datapoints, n_steps = 0, theta_trained = 0, 
                 nwalkers = 100, noise = 0.0):
        self.datapoints = datapoints
        self.n_steps = n_steps
        self.nwalkers = nwalkers
        self.noise = noise 

    def regression_model(self, datapoints, theta, mode = True, noise = 0.0):
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

    def train(self):

        datapoints = self.datapoints
        nwalkers = self.nwalkers
        xvec = self.xvec
        noise = self.noise

        regression_model = self.regression_model

        n = datapoints.shape[0]

        theta = np.ones(shape = n)
        
        x_vec = xvec(datapoints)

        par, error = walkers(nwalkers, x_vec, datapoints, regression_model
                                , theta, mov = 200, figname = None)

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