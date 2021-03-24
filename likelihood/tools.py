import numpy as np


"""
Data Science from Scratch, Second Edition, by Joel Grus (O'Reilly).Copyright 2019 Joel Grus, 978-1-492-04113-9
"""


def minibatches(dataset, batch_size, shuffle = True):

    """Generates 'batch_size'-sized minibatches from the dataset
    
    Parameters
    ----------
    dataset : List
    batch_size : int
    shuffle : bool
    
    """
    
    # start indexes 0, batch_size, 2 * batch_size, ...
    batch_starts = [start for start in range(0, len(dataset), batch_size)]
    
    if shuffle: np.random.shuffle(batch_starts)  # shuffle the batches
            
    for start in batch_starts:
        end = start + batch_size
        yield dataset[start:end]
      
        
def difference_quotient(f, x, h):
    
    """Calculates the difference quotient of 'f' evaluated at x and x + h
    
    Parameters
    ----------
    f(x) : Callable function
    x : float
    h : float
    
    Returns
    -------
    '(f(x + h) - f(x)) / h'
    
    """
    
    return (f(x + h) - f(x)) / h


def partial_difference_quotient(f, v, h):
    
    """Calculates the partial difference quotient of 'f'
    
    Parameters
    ----------
    f(x0,...,xi-th) : Callable function
    v : Vector or np.array
    h : float
    
    Returns
    -------
    the i-th partial difference quotient of f at v
    
    """
    
    w = [v_j + (h if j == i else 0)  # add h to just the ith element of v
         for j, v_j in enumerate(v)]
    return (f(w) - f(v)) / h


def estimate_gradient(f, v, h = 0.0001):
    """Calculates the gradient of 'f' at v
    
    Parameters
    ----------
    f(x0,...,xi-th) : Callable function
    v : Vector or np.array
    h : float. By default it is set to 0.0001
    
    """
    return [partial_difference_quotient(f, v, i, h)
            for i in range(len(v))]

#-------------------------------------------------------------------------

def rescale(dataset, n=1):
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
        An array containing the min of the 
        original data.
    sigma : np.array
        An array containing the (max - min) 
        of the original data.
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
        if np.max(dataset[i, :, 0]) != 0: 
            sigma.append(np.max(dataset[i, :, 0])-mu[i])
        else:
            sigma.append(1)
            
        dataset[i, :, 0] = 2*((dataset[i, :, 0] - mu[i]) / sigma[i])-1
         
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
        dataset[i, :, 0] += 1
        dataset[i, :, 0] /= 2
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


#-------------------------------------------------------------------------
