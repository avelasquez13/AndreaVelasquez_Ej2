import numpy as np
import matplotlib.pyplot as plt

x = np.array([2, -2, 5, 8, 5, 1])
y = np.array([20, -1, 12, 10, -16, 40])
z = np.array([0, 0, 0, 0, 0, 0])

t = np.array([3.23, 3.82, 2.27, 3.04, 5.65, 6.57])

sigma_t = 0.05 
v = 5

N = 10000

def likelihood(q):
    return exp(-q*q)

def loglikelihood(q):
    return -q*q

def gradient_loglikelihood(q):
    return -2*q

def leapfrog(q,p, delta_t=1E-1, niter=5):
    q_new = q
    p_new = p
    for i in range(niter):
        p_new = p_new + 0.5 * delta_t * gradient_loglikelihood(q_new)
        q_new = q_new + delta_t * p_new
        p_new = p_new + 0.5 * delta_t * gradient_loglikelihood(q_new)
    return q_new, p_new

def H(q,p):
    K = 0.5 * p * p
    U = -loglikelihood(q)
    return K + U


def MCMC(nsteps):
    q = np.zeros(nsteps)
    q[0] = np.random.normal(0,1)

    p = np.zeros(nsteps)
    p[0] = np.random.normal(0,1)
    
    for i in range(1, nsteps):
        p[i] = np.random.normal(0,1)
        q_new, p_new = leapfrog(q[i-1],p[i-1])
        E_new = H(q_new, p_new)
        E_old = H(q[i-1], p[i-1])

        alpha = min(1.0, np.exp(-E_new + E_old))
        beta = np.random.random()
        if beta < alpha:
            q[i] = q_new
        else:
            q[i] = q[i-1]

    return q

def gelman_rubin(distro, obs_data, N=10000, M=4, sigma=0.1):
    walks = {}
    for m in range(M):
        walks[m] = #Llamar a MCH sample_metropolis_hastings(distro, obs_data, N, sigma)
    
    R = np.zeros(N-1)
    for i in range(N-1):
        n = i+1
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
