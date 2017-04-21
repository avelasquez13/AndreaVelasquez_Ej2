import numpy as np
import matplotlib.pyplot as plt

x = np.array([2, -2, 5, 8, 5, 1])
y = np.array([20, -1, 12, 10, -16, 40])
z = np.array([0, 0, 0, 0, 0, 0])

t = np.array([3.23, 3.82, 2.27, 3.04, 5.65, 6.57])

sigma_t = 0.05 
v = 5

def MCH:
    

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
    
