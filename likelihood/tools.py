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
def calculate_probability(x, points = 1, cond = True):
    """Calculates the probability of the data
    
    Parameters
    ----------
    x : np.array
        An array containing the data.
    points : int
        An integer value. By default it is set to `1`.
    cond : bool
        A boolean value. By default it is set to `True`.
    
    Returns
    -------
    p : np.array
        An array containing the probability of the data.
    
    """
    
    p = []

    f = cdf(x)[0]
    for i in range(len(x)):
        p.append(f(x[i]))
    p = np.array(p)
    if cond:
        if(np.prod(p[-points]) > 1):
            print('\nThe probability of the data cannot be calculated.\n')
        else:
            if(np.prod(p[-points]) < 0):
                print('\nThe probability of the data cannot be calculated.\n')
            else:
                print('The model has a probability of {:.2f}% of being correct'.format(np.prod(p[-points])*100))
    else:
        if(np.sum(p[-points]) < 0):
            print('\nThe probability of the data cannot be calculated.\n')
        else:
            if(np.sum(p[-points]) > 1):
                print('\nThe probability of the data cannot be calculated.\n')
            else:
                print('The model has a probability of {:.2f}% of being correct'.format(np.sum(p[-points])*100))
    return p

def cdf(x, poly = 9, inv = False, plot = False):
    """Calculates the cumulative distribution function of the data

    Parameters
    ----------
    x : np.array
        An array containing the data.
    poly : int
        An integer value. By default it is set to `9`.
    inv : bool
        A boolean value. By default it is set to `False`.
    
    Returns
    -------
    cdf_ : np.array
        An array containing the cumulative distribution function.
    
    """
    
    cdf_ = np.cumsum(x) / np.sum(x)

    ox = np.sort(x)
    I = np.ones(len(ox))
    M = np.triu(I)
    df = np.dot(ox, M)
    df_ = df/np.max(df)

    if inv:
        fit = np.polyfit(df_, ox, poly)
        f = np.poly1d(fit)
    else:
        fit = np.polyfit(ox, df_, poly)
        f = np.poly1d(fit)
    
    if plot:
        if inv:
            plt.plot(df_, ox, 'o', label = 'inv cdf')
            plt.plot(df_, f(df_), 'r--', label = 'fit')
            plt.title('Quantile Function')
            plt.xlabel("Probability")
            plt.ylabel("Value")
            plt.legend()
            plt.show()
        else:
            plt.plot(ox, cdf_, 'o', label = 'cdf')
            plt.plot(ox, f(ox), 'r--', label = 'fit')
            plt.title('Cumulative Distribution Function')
            plt.xlabel("Value")
            plt.ylabel("Probability")
            plt.legend()
            plt.show()

    return f, cdf_, ox

class corr():
    """Calculates the autocorrelation of the data
    
    Parameters
    ----------
    x : np.array
        An array containing the data.
    y : np.array
        An array containing the data.
    
    Returns
    -------
    z : np.array
        An array containing the correlation of x and y.
    
    """
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.result = np.correlate(x, y, mode='full')
        self.z = self.result[self.result.size//2:]
        self.z = self.z/float(np.abs(self.z).max())
    
    def plot(self):
        plt.plot(range(len(self.z)), self.z, label = 'Correlation')
        plt.legend()
        plt.show()
    
    def __call__(self):
        return self.z

class autocorr():
    """Calculates the autocorrelation of the data
    
    Parameters
    ----------
    x : np.array
        An array containing the data.
    
    Returns
    -------
    z : np.array
        An array containing the autocorrelation of the data.
    
    """
    
    def __init__(self, x):
        self.x = x
        self.result = np.correlate(x, x, mode='full')
        self.z = self.result[self.result.size//2:]
        self.z = self.z/float(np.abs(self.z).max())
    
    def plot(self):
        plt.plot(range(len(self.z)), self.z, label = 'Autocorrelation')
        plt.legend()
        plt.show()
    
    def __call__(self):
        return self.z

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
    dataset_ = dataset.copy()
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
        dataset_[i, :] = ffilt.real
        # Calculate the period of the signal
        period = 1 / freq[L][np.argmax(fhat[L])]
        if mode:
            print(f'The {i+1}-th row of the dataset has been denoised.')
            print(f'The period is {round(period, 4)}')
    return dataset_


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
    dataset_ = dataset.copy()
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
        dataset_[i, :] += -poly
        mu.append(np.min(dataset_[i, :]))
        if np.max(dataset_[i, :]) != 0: 
            sigma.append(np.max(dataset_[i, :])-mu[i])
        else:
            sigma.append(1)
            
        dataset_[i, :] = 2*((dataset_[i, :] - mu[i]) / sigma[i])-1
         
    values = [mu, sigma, fitting]
    
    return dataset_, values

def scale(dataset, values):
    """Performs the inverse operation to the rescale function
    
    Parameters
    ----------
    dataset : np.array
        An array containing the scaled data.
    values : np.ndarray
        A set of values returned by the rescale function.
    
    """
    dataset_ = dataset.copy()
    for i in range(dataset.shape[0]):
        dataset_[i, :] += 1
        dataset_[i, :] /= 2
        dataset_[i, :] = dataset_[i, :]*values[1][i]
        dataset_[i, :] += values[0][i]
        dataset_[i, :] += values[2][i](range(dataset_.shape[1]))
    
    return dataset_

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

def RMSE(y_true, y_pred):
    """Calculates the Root Mean Squared Error
    
    Parameters
    ----------
    y_true : np.array
        An array containing the true values.
    y_pred : np.array
        An array containing the predicted values.
    
    Returns
    -------
    RMSE : float
        The Root Mean Squared Error.
    
    """

    print(f'The RMSE is {np.sqrt(np.mean((y_true - y_pred)**2))}')

    return np.sqrt(np.mean((y_true - y_pred)**2))


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

    # Calculate the autocorrelation of the data
    z = autocorr(a[0, :])
    z.plot()
    #print(z())

    N = 1000
    mu = np.random.uniform(0, 10.0)
    sigma = np.random.uniform(0.1, 1.0)
    x = np.random.normal(mu, sigma, N)
    f, cdf_, ox = cdf(x, plot = True)
    invf, cdf_, ox = cdf(x, plot = True, inv = True)