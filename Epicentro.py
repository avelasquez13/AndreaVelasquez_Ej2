import numpy as np
import matplotlib.pyplot as plt

x = np.array([2, -2, 5, 8, 5, 1])
y = np.array([20, -1, 12, 10, -16, 40])
z = np.array([0, 0, 0, 0, 0, 0])

t = np.array([3.23, 3.82, 2.27, 3.04, 5.65, 6.57])

sigma_t = 0.05
v = 5.0

N = 10000

x_c = np.array([8])
y_c = np.array([2])
z_c = np.array([0])

def likelihood(q_x, q_y, q_z):
    t_t = np.sqrt((x-q_x)**2+(y-q_y)**2+(z-q_z)**2)/v
    return np.exp(-0.5/sigma_t**2*np.sum((t-t_t)**2))

def loglikelihood(q_x, q_y, q_z):
    t_t = np.sqrt((x-q_x)**2+(y-q_y)**2+(z-q_z)**2)/v
    return -0.5/sigma_t**2*np.sum((t-t_t)**2)

def gradient_loglikelihood(q_x, q_y, q_z):
    dist = np.sqrt((x-q_x)**2+(y-q_y)**2+(z-q_z)**2)
    factor = 1/sigma_t**2 * (t-dist/v)/(v*dist)
    deriv_x = np.sum(factor*(x-q_x))
    deriv_y = np.sum(factor*(x-q_y))
    deriv_z = np.sum(factor*(x-q_z))
    return deriv_x, deriv_y, deriv_z

def leapfrog(q_x, q_y, q_z, p_x, p_y, p_z, delta_t=1E-2, niter=5):
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

def gelman_rubin(N, M=4):
    walks_x = {}
    walks_y = {}
    walks_z = {}
    for m in range(M):
        walks_x[m], walks_y[m], walks_z[m] =MCMC(N)
    
    R = np.zeros([3,N-1])
    for n in range(1, N):
        mw_x = np.zeros(M)
        mw_y = np.zeros(M)
        mw_z = np.zeros(M)
        
        vw_x = np.zeros(M)
        vw_y = np.zeros(M)
        vw_z = np.zeros(M)
        for m in range(M):
            mw_x[m] = walks_x[m][:n].mean()
            mw_y[m] = walks_y[m][:n].mean()
            mw_z[m] = walks_z[m][:n].mean()
            vw_x[m] = walks_x[m][:n].std() ** 2
            vw_y[m] = walks_y[m][:n].std() ** 2
            vw_z[m] = walks_z[m][:n].std() ** 2
        mg_x = mw_x.mean()
        mg_y = mw_y.mean()
        mg_z = mw_z.mean()
        B_x = 0.0
        B_y = 0.0 
        B_z = 0.0
        for m in range(M):
            B_x += (mw_x[m] - mg_x)**2
            B_y += (mw_y[m] - mg_y)**2
            B_z += (mw_z[m] - mg_z)**2
        B_x = n*B_x/(M-1)
        B_y = n*B_y/(M-1)
        B_z = n*B_z/(M-1)
        W_x = vw_x.mean()
        W_y = vw_y.mean()
        W_z = vw_z.mean()
    
        R[0][n-1] = (n-1)/n + (B_x/W_x)*(M+1)/(n*M)
        R[1][n-1] = (n-1)/n + (B_y/W_y)*(M+1)/(n*M)
        R[2][n-1] = (n-1)/n + (B_z/W_z)*(M+1)/(n*M)
    
    return R


R = gelman_rubin(N, M=4)

plt.plot(R[0,9000:],label='$R_x$')
plt.plot(R[1,9000:],label='$R_y$')
plt.plot(R[2,9000:],label='$R_z$')
plt.legend()
plt.savefig('GelmanRubinTest.png')
plt.close()

q_x_chain, q_y_chain, q_z_chain = MCMC(N)
plt.hist(q_x_chain, bins=200)
plt.savefig('Dist_x.png')
plt.close()

plt.hist(q_y_chain, bins=200)
plt.savefig('Dist_y.png')
plt.close()

plt.hist(q_z_chain, bins=200)
plt.savefig('Dist_z.png')
plt.close()
