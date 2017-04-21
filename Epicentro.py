import numpy as np
import matplotlib.pyplot as plt

x = np.array([2, -2, 5, 8, 5, 1])
y = np.array([20, -1, 12, 10, -16, 40])
z = np.array([0, 0, 0, 0, 0, 0])

t = np.array([3.23, 3.82, 2.27, 3.04, 5.65, 6.57])

sigma_t = 0.05
v = 5.0

N = 10000

x_c = np.array([5])
y_c = np.array([2])
z_c = np.array([0])

def likelihood(x_c, y_c, z_c):
    t_t = np.sqrt((x-x_c)**2+(y-y_c)**2+(z-z_c)**2)/v
    return np.exp(-0.5/sigma_t**2*np.sum((t-t_t)**2))

def loglikelihood(q_x, q_y, q_z):
    t_t = np.sqrt((x-x_c)**2+(y-y_c)**2+(z-z_c)**2)/v
    return -0.5/sigma_t**2*np.sum((t-t_t)**2)

def gradient_loglikelihood(q_x, q_y, q_z):
    return #TODO

def leapfrog(q_x, q_y, q_z, p_x, p_y, p_z, delta_t=0.1, niter=5):
    q_x_new = q_x 
    p_x_new = p_x

    q_y_new = q_y 
    p_y_new = p_y

    q_z_new = q_z 
    p_z_new = p_z

    for i in range(niter):
        deriv_x, deriv_y, deriv_z = gradient_loglikelihood(q_x_new, q_y_new, q_z_new)
        p_x_new = p_x_new + 0.5 * delta_t * deriv_x
        q_x_new = q_x_new + delta_t * p_x_new
        p_x_new = p_x_new + 0.5 * delta_t * deriv_x

        p_y_new = p_y_new + 0.5 * delta_t * deriv_y
        q_y_new = q_y_new + delta_t * p_y_new
        p_y_new = p_y_new + 0.5 * delta_t * deriv_y

        p_z_new = p_z_new + 0.5 * delta_t * deriv_z
        q_z_new = q_z_new + delta_t * p_z_new
        p_z_new = p_z_new + 0.5 * delta_t * deriv_z

    return q_x_new, p_x_new, q_y_new, p_y_new, q_z_new, p_z_new


def H(q_x, q_y, q_z, p_x, p_y, p_z):
    K = 0.5 * (p_x*p_x + p_y*p_y + p_z*p_z)
    U = -loglikelihood(q_x, q_y, q_z)
    return K + U


def MCMC(nsteps): 
    q_x = np.zeros(nsteps)
    q_x[0] = np.random.normal(x_c,1)

    p_x = np.zeros(nsteps)
    p_x[0] = np.random.normal(0,1)

    q_y = np.zeros(nsteps)
    q_y[0] = np.random.normal(y_c,1)

    p_y = np.zeros(nsteps)
    p_y[0] = np.random.normal(0,1)

    q_z = np.zeros(nsteps)
    q_z[0] = np.random.normal(z_c,1)

    p_z = np.zeros(nsteps)
    p_z[0] = np.random.normal(0,1)
    
    for i in range(1, nsteps):
        p_x[i] = np.random.normal(0,1)
        p_y[i] = np.random.normal(0,1)
        p_z[i] = np.random.normal(0,1)

        q_x_new, p_x_new, q_y_new, p_y_new, q_z_new, p_z_new = leapfrog(q_x[i-1], q_y[i-1], q_z[i-1], p_x[i-1], p_y[i-1], p_z[i-1])

        E_new = H(q_x_new, q_y_new, q_z_new, p_x_new, p_y_new, p_z_new)
        E_old = H(q_x[i-1], q_y[i-1], q_z[i-1], p_x[i-1], p_y[i-1], p_z[i-1])

        alpha = min(1.0, np.exp(-E_new + E_old))
        beta = np.random.random()
        if beta < alpha:
            q_x[i] = q_x_new
            q_y[i] = q_y_new
            q_z[i] = q_z_new
        else:
            q_x[i] = q_x[i-1]
            q_y[i] = q_y[i-1]
            q_z[i] = q_z[i-1]

    return q_x, q_y, q_z


def gelman_rubin(distro, obs_data, N=10000, M=4, sigma=0.1):
    walks = {}
    for m in range(M):
        walks[m] = #Llamar a MCH sample_metropolis_hastings(distro, obs_data, N, sigma)
    
    R = np.zeros(N-1)
    for n in range(1, N):
        mean_walks = np.zeros(M)
        variance_walks = np.zeros(M)
        for m in range(M):
            mean_walks[m] = walks[m][:n].mean()
            variance_walks[m] = walks[m][:n].std() ** 2
        mean_general = mean_walks.mean()    
        B = 0.0
        for m in range(M):
            B += (mean_walks[m] - mean_general)**2
        B = n*B/(M-1)
        W = variance_walks.mean()
    
        R[n-1] = (n-1)/n + (B/W)*(M+1)/(n*M)
    
    return walks, R


q_chain = MCMC(N)
plt.hist(q_chain[500:], bins=20)
plt.show()
