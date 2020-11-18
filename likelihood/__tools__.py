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
        fit = np.polyfit(xaxis, dataset[i, :, 0], n)
        weights = np.poly1d(fit)
        poly = weights(xaxis)
        fitting.append(poly)
        dataset[i, :, 0] += -fitting[i]
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
        dataset[i, :, 0] += values[2][i]
    
    return dataset

def generate_series(n, n_steps, incline = True):
    """Function that generates $n$ series of length $n_steps$
    """
    freq1, freq2, offsets1, offsets2 = np.random.rand(4, n, 1)
    
    if incline :
        slope = np.random.rand(n, 1)
    else: 
        slope = 1.0
        
    time = np.linspace(0, 1, n_steps)
    series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10)) # wave 1
    series += 0.2 * np.sin((time - offsets2) * (freq2 * 20 + 20)) # + wave 2
    series += 0.7 * (np.random.rand(n, n_steps) - 0.5) # + noise
    series += 5 * slope * time + 2 * (offsets2-offsets1) * time ** (1-offsets2)
    series = series
    return series[..., np.newaxis].astype(np.float32)