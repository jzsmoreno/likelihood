from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np
import corner


""" Función que realiza el cálculo del prior

    Argumentos:
            theta (numpy_array): arreglo de parámetros del modelo
            conditions (list): lista de 2n condiciones para el rango mín. y máx. de los n parámetros
    Return:
           lp (float): la probabilidad a priori"""

def lnprior(theta, conditions):
    try: 
        if(len(conditions) != 2*len(theta)):
            print('IndexError : Length of conditions must be twice the length of theta')
        else:
            #print(len(theta))
            cond = np.array(conditions).reshape((len(theta), 2))
            #print(cond)
            for i in range(len(theta)):
                if(cond[i, 0] < theta[i] < cond[i, 1]):
                    lp =  0.0
                else:
                    return np.inf
            return lp
    except:
        return 0.0
    
""" Función que realiza el cálculo del likelihood

    Argumentos:
            x (numpy_array): arreglo de m columnas y n filas, donde m es el número de dimensiones, 
                             n el datos por columna
            y (numpy_array): arreglo de n datos que se comparará con la salida del modelo
            model (python function): función definida a priori por el usuario que recibe dos argumentos
                                     x, theta
            theta (numpy_array): arreglo de parámetros del modelo
            conditions (list): lista de 2n condiciones para el rango mín. y máx. de los n parámetros
            var2 (float): determina el ancho del paso que dará el caminador, default = 1.0
    Return:
           lp + lhood (float): regresa el likelihood"""

def funLike(x, y, model, theta, conditions = None, var2 = 1.0):
    lp = lnprior(theta, conditions)
    inv_sigma2 = 1.0/(var2)
    y_hat = model(x, theta)
    try:
        y_hat.shape[1]
    except:
        y_hat = y_hat[np.newaxis,...].T
    lhood = 0.5*(np.sum((y-y_hat)**2*inv_sigma2 - np.log(inv_sigma2)))
    if not np.isfinite(lp):
        return np.inf
    else:
        return lp + lhood
    
""" Función que realiza la actualización de los parámetros theta

    Argumentos:
            theta (numpy_array): arreglo de parámetros del modelo
            d (float): ancho del paso Gaussiano del caminador 
    Return:
           theta_new (numpy_array): regresa el arreglo theta actualizado"""
    
def update_theta(theta, d):
    theta_new = []
    for k in range(len(theta)):
        theta_new.append(np.random.normal(theta[k], d/2.))
    return(theta_new)

""" Función que realiza la implementación del algoritmo del caminador

    Argumentos:
            x (numpy_array): arreglo de m columnas y n filas, donde m es el número de dimensiones, 
                             n el datos por columna
            y (numpy_array): arreglo de n datos que se comparará con la salida del modelo
            model (python function): función definida a priori por el usuario que recibe dos argumentos
                                     x, theta
            theta (numpy_array): arreglo de parámetros del modelo
            conditions (list): lista de 2n condiciones para el rango mín. y máx. de los n parámetros
            var2 (float): determina el ancho del paso que dará el caminador, default 0.01
            mov (int): número de movimientos que relizará el caminador, default 100
            d (float): ancho del paso Gaussiano del caminador, default 1
            tol (float): criterio de convergencia del log del likelihood, default 1.*10**-3
            mode (Bool): default True 
    Return:
           theta (numpy_array): regresa el arreglo theta actualizado
           nwalk (numpy_array): actualizaciones del arreglo de theta para cada uno de los movimientos efectuados por el caminador
           y0 (float): valor del logaritmo del likelihood"""
    
def walk(x, y, model, theta, conditions = None, var2 = 0.01, mov = 100, d = 1, tol = 1.*10**-3, mode = True):
    greach = False
    nwalk = []
    for i in range(mov):
        nwalk.append(theta)
        theta_new = update_theta(theta, d)
        if not (greach):
            y0=funLike(x, y, model, theta, conditions, var2)
            y1=funLike(x, y, model, theta_new, conditions, var2)
            if(y0 <= tol):
                if(mode):
                    print('Goal Reached')
                    greach = True
                    return(theta, nwalk, y0)
            else:
                if(y1 <= tol):
                    if(mode):
                        print('Goal Reached')
                        greach = True
                        return(theta_new, nwalk, y1)
                else:
                    ratio=y0/y1
                    boltz = np.random.rand(1)
                    prob = np.exp(-ratio)
                    if(y1<y0):
                        #print('Accepted')
                        theta = theta_new
                        theta_new = update_theta(theta, d)
                    else:
                        if(prob>boltz):
                            #print('Accepted')
                            theta = theta_new
                            theta_new = update_theta(theta, d)
                        else:
                            #print('Non-Accepted')
                            theta_new = update_theta(theta, d)
    if(mode):  
        print('Max. number of iterations have been reached! ,', 'The log likelihood is :', y0)
        #clear_output(wait = True)
    return theta, nwalk, y0

""" Función que realiza la implementación del algoritmo de múltiples caminadores

    Argumentos:
            
            x (numpy_array): arreglo de m columnas y n filas, donde m es el número de dimensiones, 
                             n el datos por columna
            y (numpy_array): arreglo de n datos que se comparará con la salida del modelo
            model (python function): función definida a priori por el usuario que recibe dos argumentos
                                     x, theta
            theta (numpy_array): arreglo de parámetros del modelo
            conditions (list): lista de 2n condiciones para el rango mín. y máx. de los n parámetros
            var2 (float): determina el ancho del paso que dará el caminador, default 0.01
            mov (int): número de movimientos que relizará el caminador, default 100
            d (float): ancho del paso Gaussiano del caminador, default 1
            tol (float): criterio de convergencia del log del likelihood, default 1.*10**-3
            mode (Bool): default False, parámetro que indica que estaremos trabajando con más de un caminador 
    Return:
           par (numpy_array): regresa el arreglo theta que encontró cada uno de los nwalkers caminadores,
           error (numpy_array): es el logaritmo del likelihood"""

def walkers(nwalkers, x, y, model, theta, conditions = None, var2 = 0.01, mov = 100, d = 1, tol = 1.*10**-3, mode = False):
    result = []
    error = []
    par = []
    for i in range(nwalkers):
        theta, nwalk, y0 = walk(x, y, model, theta, conditions, var2, mov, d, tol, mode)
        par.append(theta)
        nwalk = np.array(nwalk).reshape((len(nwalk), len(nwalk[i])))
        error.append(y0)
        for k in range(nwalk.shape[1]):
            sub = '$\\theta _{'+str(k)+'}$'
            plt.plot(range(len(nwalk[:, k])), nwalk[:, k], '-', label = sub)
            plt.ylabel('$\\theta$')
            plt.xlabel('iterations')
            #plt.legend()
    plt.show()
    if(nwalk.shape[1] == 2):
        fig = corner.hist2d(nwalk[:, 0],nwalk[:, 1], range=None, bins=18, 
                    smooth=True,plot_datapoints=True,
                    plot_density=True)
        plt.ylabel('$\\theta_{1}$')
        plt.xlabel('$\\theta_{0}$')
        plt.savefig("1401.png")
    return(par, error)

