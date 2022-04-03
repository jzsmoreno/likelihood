import numpy as np
import matplotlib.pyplot as plt

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


def partial_difference_quotient(f, v, i, h):
    
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

def fft_denoise(dataset, sigma = 0, mode = True):
    """Performs the noise removal using the Fast Fourier Transform
    
    Parameters
    ----------
    dataset : np.array
        An array containing the noised data.
    sigma : float
        A float between 0 and 1. By default it is set to `0`.
    mode : bool
        A boolean value. By default it is set to `True`.
    
    Returns
    -------
    dataset : np.array
        An array containing the denoised data.
    
    """

    for i in range(dataset.shape[0]):
        n = dataset.shape[1]
        fhat = np.fft.fft(dataset[i, :], n)
        freq = (1/n) * np.arange(n)         
        L = np.arange(1,np.floor(n/2),dtype='int')
        PSD = fhat * np.conj(fhat) / n
        indices = PSD > np.mean(PSD) + sigma * np.std(PSD)
        PSDclean = PSD * indices  # Zero out all others
        fhat = indices * fhat   
        ffilt = np.fft.ifft(fhat) # Inverse FFT for filtered time signal
        dataset[i, :] = ffilt.real
        # Calculate the period of the signal
        period = 1 / freq[L][np.argmax(fhat[L])]
        if mode:
            print(f'The {i+1}-th row of the dataset has been denoised.')
            print(f'The period is {period}')
    return dataset


def feature_importance(dataset, values):
    """Calculates the importance of each feature
    
    Parameters
    ----------
    dataset : np.array
        An array containing the scaled data.
    values : np.ndarray
        A set of values returned by the linear function.
    
    Returns
    -------
    importance : np.array
        An array containing the importance of each feature.
    
    """
    
    importance = []
    print('\nFeature importance:')
    U, S, VT = np.linalg.svd(dataset, full_matrices=False)
    w = (VT.T@np.linalg.inv(np.diag(S))@U.T).T@values

    for i in range(dataset.shape[0]):
        a = np.around(w[i], decimals=4)
        importance.append(a)
        print(f'The importance of the {i+1} feature is {a}')
    return np.array(importance)

def cal_average(y, alpha = 1):
    """Calculates the moving average of the data
    
    Parameters
    ----------
    y : np.array
        An array containing the data.
    alpha : float
        A float between 0 and 1. By default it is set to `1`.
    
    Returns
    -------
    average : float
        The average of the data.
    
    """
    
    n = int(alpha * len(y))
    w = np.ones(n) / n
    average = np.convolve(y, w, mode='same') / np.convolve(np.ones_like(y), w, mode='same')
    return average

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
            fit = np.polyfit(xaxis, dataset[i, :], n)
            f = np.poly1d(fit)
            poly = f(xaxis)
            fitting.append(f)
        else:
            fitting.append(0.0)
        dataset[i, :] += -poly
        mu.append(np.min(dataset[i, :]))
        if np.max(dataset[i, :]) != 0: 
            sigma.append(np.max(dataset[i, :])-mu[i])
        else:
            sigma.append(1)
            
        dataset[i, :] = 2*((dataset[i, :] - mu[i]) / sigma[i])-1
         
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
        dataset[i, :] += 1
        dataset[i, :] /= 2
        dataset[i, :] = dataset[i, :]*values[1][i]
        dataset[i, :] += values[0][i]
        dataset[i, :] += values[2][i](range(dataset.shape[1]))
    
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
    return series.astype(np.float32)


#-------------------------------------------------------------------------
if __name__ == '__main__':
    # Generate data
    x = np.random.rand(3, 100)
    y = 0.1*x[0, :] + 0.4*x[1, :] + 0.5*x[2, :]
    importance = feature_importance(x, y)

    a = generate_series(1, 40, incline=False)
    # Graph the data for visualization
    plt.plot(range(len(a[0, :])), a[0, :], label = 'Original Data')
    plt.legend()
    plt.xlabel('Time periods')
    plt.ylabel('$y(t)$')
    plt.show()

    a_denoise = fft_denoise(a)
    
    plt.plot(range(len(a_denoise[0, :])), a_denoise[0, :], label = 'Denoise Data')
    plt.legend()
    plt.xlabel('Time periods')
    plt.ylabel('$y(t)$')
    plt.show()