import numpy as np
import matplotlib.pyplot as plt


T = 2000  # Total time in ms
dt = 0.04  # Integration time step in ms
nt = round(T/dt)  # Time steps
N = 10  # Number of neurons

# Izhikevich Parameters
C = 250  # capacitance
vr = -60  # resting membrane
b = -2  # resonance parameter
ff = 2.5  # k parameter for Izhikevich, gain on v
vpeak = 30  # peak voltage
vreset = -65  # reset voltage
vt = vr + 40 - (b/ff)  # threshold
u = np.zeros((N, 1))  # initialize adaptation
a = 0.01  # adaptation reciprocal time constant
d = 200  # adaptation jump current
tr = 2  # synaptic rise time
td = 20  # decay time
p = 0.1  # sparsity
G = 5 * 10**3  # Gain on the static matrix with 1/sqrt(N) scaling weights. Note that the units of this have to be in pA.
Q = 5 * 10**3  # Gain on the rank-k perturbation modified by RLS. Note that the units of this have to be in pA
Irh = 0.25 * ff * (vt - vr)**2

# Storage variables for synapse integration
IPSC = np.zeros((N, 1))  # post synaptic current
h = np.zeros((N, 1))
r = np.zeros((N, 1))
hr = np.zeros((N, 1))
JD = np.zeros((N, 1))


#-----Initialization---------------------------------------------
v = vr + (vpeak - vr) * np.random.rand(N, 1)  # initial distribution
v_ = v  # These are just used for Euler integration, previous time step storage
np.random.seed(1)
## Target signal  COMMENT OUT TEACHER YOU DONT WANT, COMMENT IN TEACHER YOU WANT.
zx = np.sin(2 * 5 * np.pi * np.arange(1, nt+1) * dt / 1000)
# print(zx.shape)
##
k = min(zx.shape)  # used to get the dimensionality of the approximant correctly.  Typically will be 1 unless you specify a k-dimensional target function.
print(f"k={k}")

OMEGA = G * (np.random.randn(N, N)) * (np.random.rand(N, N) < p) / (p * np.sqrt(N))  # Static weight matrix.
z = np.zeros((k, 1))  # initial approximant
BPhi = np.zeros((N, k))  # initial decoder.  Best to keep it at 0.
tspike = np.zeros((5 * nt, 2))  # If you want to store spike times,
ns = 0  # count total number of spikes
BIAS = 1000  # Bias current, note that the Rheobase is around 950 or something.  I forget the exact formula for this but you can test it out by shutting weights and feeding constant currents to neurons
E = (2 * np.random.rand(N, k) - 1) * Q  # Weight matrix is OMEGA0 + E*BPhi';
##
Pinv = np.eye(N) * 2  # initial correlation matrix, coefficient is the regularization constant as well
step = 20  # optimize with RLS only every 50 steps
imin = round(500 / dt)  # time before starting RLS, gets the network to chaotic attractor
icrit = round(1500 / dt)  # end simulation at this time step
current = np.zeros((nt, k))  # store the approximant
RECB = np.zeros((nt, 5))  # store the decoders
REC = np.zeros((nt, 10))  # Store voltage and adaptation variables for plotting
i = 0

# SIMULATION

ilast = i
# icrit = ilast  # uncomment this, and restart cell if you want to test
# performance before icrit.
for i in range(ilast, nt+1):
    # EULER INTEGRATE
    I = IPSC + E @ z + BIAS  # postsynaptic current
    v = v + dt*((ff*(v-vr)*(v-vt) - u + I))/C  # v(t) = v(t-1)+dt*v'(t-1)
    u = u + dt*(a*(b*(v_-vr)-u))  # same with u, the v_ term makes it so that the integration of u uses v(t-1), instead of the updated v(t)
    # index = np.where(v >= vpeak)
    print(np.max(v))
    index = np.where(np.greater_equal(v, vpeak))[0]
    # print(f"index={index}")
    if len(index) > 0:
        JD = np.sum(OMEGA[:, index], axis=1)  # compute the increase in current due to spiking
        tspike[ns:ns+index[0].size, :] = np.column_stack((index[0], np.zeros(index[0].size)+dt*i))  # uncomment this
        # if you want to store spike times. Takes longer.
        ns = ns + index.size

    #synapse for single exponential
    if tr == 0:
        IPSC = IPSC * np.exp(-dt/td) + JD * (len(index) > 0) / td
        r = r * np.exp(-dt/td) + (v >= vpeak) / td
    else:
    #synapse for double exponential
        IPSC = IPSC * np.exp(-dt/tr) + h * dt
        h = h * np.exp(-dt/td) + JD * (max(np.shape((index))) > 0) / (tr * td)  #Integrate the current
        r = r * np.exp(-dt/tr) + hr * dt
        hr = hr * np.exp(-dt/td) + (v >= vpeak) / (tr * td)
    z = np.matmul(BPhi.T, r) #approximant
    err = z - zx[i] #error
    ## RLS
    if i % step == 1:
        if i > imin:
            if i < icrit:
                cd = np.dot(Pinv, r)
                BPhi = BPhi - np.dot(cd, err.T)
                Pinv = Pinv - np.dot(cd, cd.T) / (1 + np.dot(r.T, cd))


    u = u + d*(v>=vpeak)
    v = v + (vreset-v)*(v>=vpeak)
    v_ = v
    REC[i,:] = np.concatenate((v[0:5, 0], u[0:5, 0]), axis=0)
    current[i,:] = z[:, 0]
    RECB[i,:] = BPhi[0:5, i]

    # if i % round(100/dt) == 1:
    #     plt.draw()
    #     gg = max(1, i - round(3000/dt))
    #     plt.figure(2)
    #     plt.plot(dt*(np.arange(gg, i+1))/1000, zx[gg:i+1], 'k', linewidth=2)
    #     plt.plot(dt*(np.arange(gg, i+1))/1000, current[gg:i+1,:], 'b--', linewidth=2)
    #     plt.xlabel('Time (s)')
    #     plt.ylabel(r'$\hat{x}(t)$')
    #     plt.legend(['Approximant', 'Target Signal'])
    #     plt.xlim(np.array([dt*i-3000, dt*i])/1000)
    #     plt.figure(3)
    #     plt.plot((np.arange(0, i+1))*dt/1000, RECB[0:i+1,:])
    #     plt.figure(14)
    #     plt.plot(tspike[0:ns,1], tspike[0:ns,0], 'k.')
    #     plt.ylim([0, 100])

    tspike = tspike[tspike[:,1] != 0,:]
    M = tspike[tspike[:,1] > dt*icrit]
    AverageFiringRate = 1000*len(M)/(N*(T-dt*icrit))

plt.figure(30)
for j in range(1, 6):
    plt.plot((np.arange(1, i+1))*dt/1000, REC[0:i+1,j-1]/(vpeak-vreset)+j)
plt.xlim([T/1000-2, T/1000])
plt.xlabel('Time (s)')
plt.ylabel('Neuron Index')
plt.title('Post Learning')

plt.figure(31)
for j in range(1, 6):
    plt.plot((np.arange(1, i+1))*dt/1000, REC[0:i+1,j-1]/(vpeak-vreset)+j)
plt.xlim([0, imin*dt/1000])
plt.xlabel('Time (s)')
plt.ylabel('Neuron Index')
plt.title('Pre-Learning')

plt.figure(40)
Z = np.linalg.eigvals(OMEGA+E*BPhi.T)
Z2 = np.linalg.eigvals(OMEGA)
plt.plot(Z2, 'r.')