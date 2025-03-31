"""In this module I simulate a SRNN with arbitrary input signals. The models used 
are izhikevich and theta function.
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class Izhikevich:
    def __init__(self, supervisor:np.ndarray, C:float=250, v_r:float=-60, v_t:float=-20, b:float=0,
                 v_peak:float=35, v_reset:float=-65, a:float=0.01, d:float=200,
                 I_BIAS:float=1000, k:float=2.5, tau_r:float=2, tau_d:float=20,
                 dt:float=.04, T:float=5e3, g:float=3e3, N:int=1000, p:float=1.0, 
                 l:float=2, Q:float=5e3) -> None:
        """Initialize the system with corresponding parameters.

        Parameters
        ----------
        C : float, optional
            Capacitance of neurons in [mu F], by default 250
        v_r : float, optional
            Resting membrane potential in [mV], by default -60
        v_t : float, optional
            Threshold voltage in [mV], by default -20
        b : float, optional
            Resonance parameter in [nS], by default 0
        v_peak : float, optional
            Peak voltage where the neuron spikes in [mV], by default 35
        v_reset : float, optional
            Voltage for the spiking neuron to reset to in [mV], by default -65
        a : float, optional
            Adaptation Reciprocal time constant in [ms^-1], by default 0.01
        d : float, optional
            Adaptation jump current in [pA], by default 200
        I_BIAS : float, optional
            Bias current input to all neurons for them to enter the
            near spike regime in [pA], by default 1000
        k : float, optional
            Gain on voltage `v` in [ns/mV], by default 2.5
        tau_r : float, optional
            Synaptic rise time constant in [ms], by default 2
        tau_d : float, optional
            Synaptic decay time constant in [ms], by default 20
        dt : float, optional
            Time step of the simulation im [ms], by default 1e-4
        T : float, optional
            Duration of the simulation in [ms], by default 5e3
        g : float, optional
            Gain on the static weight matrix, by default 3000
        N : int, optional
            Number of neurons in SRNN, by default 1000
        p : float, optional
            Sparsity coefficient, between 0 and 1 where 1 means fully 
            connected, by default 1.0
        Q : float, optional
            Gain on the rank-k perturbation modified by RLS.
            Note that the units of this have to be in [pA], by default 5e3
        l : float, optional
            Ridge Regularization parameter lambda, by default 2.0
        """
        self.C = C
        self.v_r = v_r
        self.v_t = v_t
        self.b = b
        self.v_peak = v_peak
        self.v_reset = v_reset
        self.a = a
        self.d = d
        self.I_BIAS = I_BIAS
        self.k = k
        self.tau_r = tau_r
        self.tau_d = tau_d
        self._dt = dt
        self._T = T
        self.time = np.arange(0, T, dt)
        self._N = N
        self._G = g
        self._Q = Q
        self._p = p
        self._w = self._G * np.random.randn(N, N) \
            * (np.random.rand(N, N)<p) / (p*np.sqrt(N))    # weight matrix. Constant for now.
        self.v = v_r + (v_peak - v_r) * np.random.rand(N, 1)    # Voltage of all neurons
        self.u = np.zeros(shape=(N, 1), dtype=float)    # Adaptation parameter
        self.s = np.zeros(shape=(N, 1), dtype=float)    # Synaptic filter
        self.r = np.zeros(shape=(N, 1), dtype=float)    # spike rate
        self.h = np.zeros(shape=(N, 1), dtype=float)    # Synaptic filter
        self.wh = np.zeros(shape=(N, 1), dtype=float)   # w @ h
        # Record the spike times of the system
        self.tspike = []
        for i in range(self._N):
            self.tspike.append([])
        # Configs for RLS
        self.l = l
        self.supervisor = supervisor
        dim = min(supervisor.shape)
        self.phi = np.zeros(shape=(self._N, dim), dtype=float)
        self.eta = (2 * np.random.rand(N, dim) - 1) * Q
        self.Pinv = np.eye(self._N) * self.l
        self.x_hat = self.phi.T @ self.r
        self.x_hat_rec = np.zeros((self.time.size, self.x_hat.size), dtype=float)
        self.ipsc = np.zeros(shape=(N, 1), dtype=float)
        
    def _calc_ipsc(self):
        """Calculate the total post-synaptic current.
        It is a combination of bias current, the encoding of the output 
        and the synaptic potential of the network.
        """
        self.ipsc = self.I_BIAS + self.s + self.eta @ self.x_hat
    
    def _v_dot(self):
        self._calc_ipsc()   # update the post-synaptic current
        return (self.k * (self.v - self.v_r) * (self.v - self.v_t) \
            - self.u + self.ipsc) / self.C
        
    def _u_dot(self):
        return self.a * (self.b * (self.v - self.v_r) - self.u)
    
    def _r_dot(self):
        return -self.r / self.tau_d + self.h

    def _h_dot(self):
        return -self.h / self.tau_r
    
    def _wh_dot(self):
        return - self.wh / self.tau_r
    
    def render(self, rls_start, rls_stop, rls_step, n_neurons):
        """Simulate the system.
        Randomly choose a neuron and return its voltage trace.
        And return its spike times.
        """
        if n_neurons > self._N:
            raise ValueError(f"n_neurons = {n_neurons} should be less than or equal to {self._N}")
        random_neuron = np.random.choice(a=self._N, size=n_neurons, replace=False)
        voltage_trace = np.zeros(shape=(self.time.size, n_neurons), dtype=float)
        # Setup for RLS
        rls_start = int(rls_start // self._dt)
        rls_stop = int(rls_stop // self._dt)
        # Rendering loop
        for i in tqdm(range(len(self.time))):
            v_new = self.v + self._dt * self._v_dot()
            self.u += self._dt * self._u_dot()
            self.v = v_new
            self.r += self._dt * self._r_dot()
            self.h += self._dt * self._h_dot()
            self.wh += self._dt * self._wh_dot()

            # Set the spiking rules
            spike_mask = (self.v >= self.v_peak)
            self.v[spike_mask] = self.v_reset
            self.u[spike_mask] += self.d
            self.h[spike_mask] += 1 / (self.tau_d * self.tau_r)
            
        
            if True in spike_mask:
                spiking_neurons = np.where(spike_mask)[0]
                JD = np.sum(self._w[:, spiking_neurons], axis=1).reshape(-1, 1)
                # print(JD.shape)
                self.wh += JD / (self.tau_r * self.tau_d)
                for neuron in spiking_neurons:
                    self.tspike[neuron].append(self.time[i])

            self.s = self.s * np.exp(-self._dt / self.tau_d) + self.wh * self._dt

            # RLS for FORCE
            self.x_hat = self.phi.T @ self.r
            self.x_hat_rec[i] = self.x_hat.flatten()
            
            if i > rls_start and i < rls_stop:
                if i % rls_step == 0:
                    self.rls(i)
            
            # record
            voltage_trace[i] = self.v[random_neuron, 0]

        
        return voltage_trace
    
    def rls(self, i):
        """Run the system with the force method. Return the final decoder
        """
        error = self.x_hat - self.supervisor[i].reshape(-1, 1)
        q = self.Pinv @ self.r
        self.Pinv -= (q @ q.T) / (1 + self.r.T @ q)
        self.phi -= (q @ error.T)
        
        
        

def z_transform(signal):
    return (signal - signal.mean(axis=0)) / signal.std(axis=0)

def main():
    """Main body to test the code
    """
    from test_models import LorenzAttractor
    
    np.random.seed(2023)
    
    T = 15000
    dt = 4e-2
    t = np.arange(0, T, dt)

    x = np.sin(2 * 5 * np.pi * t / 1000)
    x = x.reshape(-1, 1)

    signal = z_transform(x)

    plt.plot(t, signal[:, 0], lw=.5)
    plt.show()
    
    params = {
        "N": 2000,
        "C": 250,
        "v_r": -60,
        "v_t": -60 + 40 - (-2 / 2.5), # v_t = v_r + 40 - (b / k)
        "k": 2.5,
        "b": -2,
        "v_peak": 30,
        "v_reset": -65,
        "a": 0.01,
        "d": 200,
        "tau_r": 2,
        "tau_d": 20,
        "p": 0.1,
        "g": 5e3,
        "Q": 5e3,
        "I_BIAS": 1000,
        "l": 2,       
    }
    model = Izhikevich(supervisor=signal, T=T, dt=dt, **params)
    voltage_trace = model.render(rls_start=5000, rls_stop=10000, rls_step=20, n_neurons=10)
    # plot the decoded signal vs the original input
    plt.plot(t, signal[:, 0], lw=.5)
    plt.plot(t, model.x_hat_rec[:, 0], lw=.5)
    plt.savefig("img/sin/output.jpg", dpi=200, bbox_inches="tight")
    plt.show()
    
    # plot the spikes
    plt.eventplot(model.tspike)
    plt.savefig("img/sin/rasterplot.jpg", dpi=200, bbox_inches="tight")
    plt.show()
    
    # plot the voltage trace of random neurons
    fig = plt.figure(31)
    ax = fig.add_subplot()
    for i in range(voltage_trace.shape[-1]):
        signal = voltage_trace[:, i]
        minim = np.min(signal)
        maxim = np.max(signal)
        signal = (signal - minim) / (maxim - minim) + i
        ax.plot(t, signal)
    plt.savefig("img/sin/voltage_trace.jpg", dpi=200, bbox_inches="tight")
    plt.show()
    

if __name__ == "__main__":
    main()
                