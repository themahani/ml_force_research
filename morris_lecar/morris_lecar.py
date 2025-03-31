"""Implementation of Reservoir machine learning with FORCE training using the Morris Lecar model.
We will be abiding by Dale's law stating that each neuron can either have excitatory or inhibitory outgoing synapses.
"""

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm



class MorrisLecar:
    def __init__(self, supervisor:np.ndarray, dt:float, T:float, 
                 N:int=1000, BIAS:float=100, C:float=20, g_L:float=2, g_K:float=8, 
                 g_Ca:float=4.4, E_L:float=-60, E_K:float=-84, E_Ca:float=120, v1:float=-1.2,
                 v2:float=18, v3:float=2, v4:float=30, phi:float=0.04, a_r:float=1.1, 
                 a_d:float=0.19, v_t:float=2, k_p:float=5, t_max:float=1.0,
                 E_AMPA:float=-70, E_GABA:float=-70, Q:float=5e3, l:float=2.0) -> None:
        # Morris Lecar model parameters
        self._N = N
        self._BIAS = BIAS
        self._C = C
        self._g_L = g_L
        self._g_K = g_K
        self._g_Ca = g_Ca
        self._E_L = E_L
        self._E_K = E_K
        self._E_Ca = E_Ca
        self._v1 = v1
        self._v2 = v2
        self._v3 = v3
        self._v4 = v4
        self._phi = phi
        self._v_t = v_t
        self._a_r = a_r
        self._a_d = a_d
        self._k_p = k_p
        self._t_max = t_max
        self._E_AMPA = E_AMPA
        self._E_GABA = E_GABA
        
        # Network connections
        self.v = np.zeros(shape=(N, 1), dtype=float)
        self.w = np.random.normal(loc=0, scale=1/np.sqrt(N)) * 50
        self.s = np.zeros(shape=(N, 1), dtype=float)
        self.n = np.zeros(shape=(N, 1), dtype=float)
        self.E = np.random.choice(a=[self._E_AMPA, self._E_GABA], size=(N, ))
        # Let's follow Dale's law
        self.E = np.tile(self.E, reps=(N, 1))
        
        # Time series
        self._dt = dt
        self._duration = T
        self.time = np.arange(0, T, dt)
        self._nt = self.time.size
        self.exp = None
        # Encoding and decoding
        self.l = l
        self.sup = supervisor
        dim = min(self.sup.shape)
        self.dec = np.zeros(shape=(N, dim), dtype=float)
        self.eta = Q * (2 * np.random.rand(N, dim) - 1)
        self.Pinv = np.eye(self._N) * self.l
        self.x_hat = self.dec.T @ self.s
        self.x_hat_rec = np.zeros((self.time.size, self.x_hat.size), dtype=float)
        self.ipsc = np.zeros(shape=(N, 1), dtype=float)
        
    def m_ss(self) -> np.ndarray:
        return 0.5 * (1 + np.tanh((self.v - self._v1) / self._v2))    
    
    def n_ss(self) -> np.ndarray:
        return 0.5 * (1 + np.tanh((self.v - self._v3) / self._v4))
    
    def tau_n(self) -> np.ndarray:
        return 1 / (self._phi * np.cosh((self.v - self._v3) / (2 * self._v4)))
    
    def T(self):
        self.exp = self._t_max / (1 + np.exp(-(self.v - self._v_t) / self._k_p))
        return self._t_max / (1 + np.exp(-(self.v - self._v_t) / self._k_p))
        
    def s_dot(self):
        return self._a_r * self.T() * (1 - self.s) - self._a_d * self.s
    
    def n_dot(self):
        return (self.n_ss() - self.n) / self.tau_n()
    
    def calc_ipsc(self) -> None:
        self.ipsc = (self.w * (self.v - self.E)) @ self.s
        
    def v_dot(self):
        self.calc_ipsc()    # Calculate the new post-synaptic potential
        
        return (self._BIAS - self._g_L * (self.v - self._E_L) \
            - self._g_K * self.n * (self.v - self._E_K) \
            - self._g_Ca * self.m_ss() * (self.v - self._E_Ca) \
            + self.ipsc) / self._C
    
    def render(self, rls_start, rls_stop, rls_step, n_neurons:int=10):
        random_neuron = np.random.choice(a=self._N, size=n_neurons, replace=False)
        voltage_trace = np.zeros(shape=(self.time.size, n_neurons), dtype=float)
        exp_trace = np.zeros((self.time.size, self._N))
        # Setup for RLS
        rls_start = int(rls_start // self._dt)
        rls_stop = int(rls_stop // self._dt)
        
        for i in tqdm(range(self._nt)):
            dv = self._dt * self.v_dot()
            self.n += self._dt * self.n_dot()
            self.s += self._dt * self.s_dot()
            self.v += dv
            
            exp_trace[i] = self.exp[random_neuron, 0]
            voltage_trace[i] = self.v[random_neuron, 0]
            
            self.x_hat = self.dec.T @ self.s
            self.x_hat_rec[i] = self.x_hat.flatten()
        
            if i > rls_start and i < rls_stop:
                if i % rls_step == 0:
                    self.rls(i)
        
        return random_neuron, voltage_trace, exp_trace
    
    def rls(self, i):
        """Run the system with the force method. Return the final decoder
        """
        error = self.x_hat - self.sup[i].reshape(-1, 1)
        q = self.Pinv @ self.s
        self.Pinv -= (q @ q.T) / (1 + self.s.T @ q)
        self.dec -= (q @ error.T)
        

class MorrisLecarEI:
    def __init__(self, supervisor:np.ndarray, dt:float, T:float, 
                 NE:int=10, NI:int=10, I_e:float=90, I_i:float=90, C:float=20, g_L:float=2, g_K:float=8, 
                 g_Ca:float=4.4, E_L:float=-60, E_K:float=-84, E_Ca:float=120, v1:float=-1.2,
                 v2:float=18, v3:float=2, v4:float=30, phi:float=0.04, a_r:float=1.1, 
                 a_d:float=0.19, v_t:float=2, k_p:float=5, t_max:float=1.0,
                 E_AMPA:float=0, E_GABA:float=-75, Q:float=5, l:float=2.0, gbar:float=1) -> None:
        # Morris Lecar model parameters
        self._N = NE + NI
        self._NE = NE
        self._NI = NI
        self._I_e = I_e
        self._I_i = I_i
        self._C = C
        self._g_L = g_L
        self._g_K = g_K
        self._g_Ca = g_Ca
        self._E_L = E_L
        self._E_K = E_K
        self._E_Ca = E_Ca
        self._v1 = v1
        self._v2 = v2
        self._v3 = v3
        self._v4 = v4
        self._phi = phi
        self._v_t = v_t
        self._a_r = a_r
        self._a_d = a_d
        self._k_p = k_p
        self._t_max = t_max
        self._E_AMPA = E_AMPA
        self._E_GABA = E_GABA
        self.gbar = gbar
        
        # time series
        self.ve = -50 + 30 * np.random.uniform(size=(NE, 1))
        self.se = np.zeros(shape=(NE, 1), dtype=float)
        self.ne = np.zeros(shape=(NE, 1), dtype=float)
        self.vi = -50 + 30 * np.random.uniform(size=(NI, 1))
        self.si = np.zeros(shape=(NI, 1), dtype=float)
        self.ni = np.zeros(shape=(NI, 1), dtype=float)
        # Threshold potential for 
        # self.E = np.ones(shape=(self._N, 1))
        # self.E[:NE, 0] *= E_AMPA     # first half of the neurons are excitatory
        # self.E[NE:, 0] *= E_GABA     # second half is inhibitory
        
        # Network connections
        scale = 1 / np.sqrt(self._N)
        # self.w = np.random.uniform(low=0, high=1, size=(N, N)) * scale
        self.wee = np.random.rand(NE, NE) < 0.5
        self.wie = np.random.rand(NI, NE) < 0.5
        self.wei = np.random.rand(NE, NI) < 0.5
        self.wii = np.random.rand(NI, NI) < 0.5
        
        # Time series
        self._dt = dt
        self._duration = T
        self.time = np.arange(0, T, dt)
        self._nt = self.time.size
        self.exp = None
        # Encoding and decoding
        self.l = l
        self.sup = supervisor
        dim = np.min(self.sup.shape)
        # self.dec = np.zeros(shape=(N, dim), dtype=float)
        # self.eta = Q * (2 * np.random.rand(N, dim) - 1)
        # self.Pinv = np.eye(self._N) * self.l
        # self.x_hat = self.dec.T @ self.s
        # self.x_hat_rec = np.zeros((self.time.size, self.x_hat.size), dtype=float)
        self.ipsce = np.zeros(shape=(NE, 1), dtype=float)
        self.ipsci = np.zeros(shape=(NI, 1), dtype=float)
        
    def m_ss(self, v) -> np.ndarray:
        return 0.5 * (1 + np.tanh((v - self._v1) / self._v2))    
    
    def n_ss(self, v) -> np.ndarray:
        return 0.5 * (1 + np.tanh((v - self._v3) / self._v4))
    
    def tau_n(self, v) -> np.ndarray:
        return 1 / (self._phi * np.cosh((v - self._v3) / (2 * self._v4)))
    
    def T(self, v):
        exp = np.exp(-(v - self._v_t) / self._k_p)
        return self._t_max / (1 + exp)
        
    def s_dot(self, s, v):
        return self._a_r * self.T(v) * (1 - s) - self._a_d * s
    
    def n_dot(self, n, v):
        return (self.n_ss(v) - n) / self.tau_n(v)
    
    def calc_ipsc(self) -> None:
        # Exc population
        ipscee = -self.gbar * (self.wee @ self.se) * (self.ve - self._E_AMPA)
        ipscei = -self.gbar * (self.wei @ self.si) * (self.ve - self._E_GABA)
        self.ipsce = ipscee + ipscei
        # Inh population
        ipscie = -self.gbar * (self.wie @ self.se) * (self.vi - self._E_AMPA)
        ipscii = -self.gbar * (self.wii @ self.si) * (self.vi - self._E_GABA)
        self.ipsci = ipscie + ipscii
        
    def v_dot(self):
        self.calc_ipsc()    # Calculate the new post-synaptic potential
        # exc population
        I_Le = -self._g_L * (self.ve - self._E_L)
        I_Ke = -self._g_K * self.ne * (self.ve - self._E_K)
        I_Cae = -self._g_Ca * self.m_ss(self.ve) * (self.ve - self._E_Ca)
        ve_dot = (self._I_e + I_Le + I_Ke + I_Cae + self.ipsce) / self._C
        # inh population
        I_Li = -self._g_L * (self.vi - self._E_L)
        I_Ki = -self._g_K * self.ni * (self.vi - self._E_K)
        I_Cai = -self._g_Ca * self.m_ss(self.vi) * (self.vi - self._E_Ca)
        vi_dot = (self._I_i + I_Li + I_Ki + I_Cai + self.ipsci) / self._C
        return [ve_dot, vi_dot]

    def render(self, rls_start, rls_stop, rls_step):
        voltage_trace = np.zeros(shape=(self.time.size, self._N), dtype=float)
        
        # Setup for RLS
        rls_start = int(rls_start // self._dt)
        rls_stop = int(rls_stop // self._dt)
        
        for i in tqdm(range(self._nt)):
            ve_dot, vi_dot = self.v_dot()
            
            dve = self._dt * ve_dot
            self.ne += self._dt * self.n_dot(self.ne, self.ve)
            self.se += self._dt * self.s_dot(self.se, self.ve)
            self.ve += dve
            
            dvi = self._dt * vi_dot
            self.ni += self._dt * self.n_dot(self.ni, self.vi)
            self.si += self._dt * self.s_dot(self.si, self.vi)
            self.vi += dvi
            
            voltage_trace[i] = np.concatenate([self.ve[:, 0], self.vi[:, 0]])

            # self.x_hat = self.dec.T @ self.s
            # self.x_hat_rec[i] = self.x_hat.flatten()
        
            # if i > rls_start and i < rls_stop:
            #     if i % rls_step == 0:
            #         self.rls(i)
        
        return voltage_trace
    
    # def rls(self, i):
    #     """Run the system with the force method. Return the final decoder
    #     """
    #     error = self.x_hat - self.sup[i].reshape(-1, 1)
    #     q = self.Pinv @ self.s
    #     self.Pinv -= (q @ q.T) / (1 + self.s.T @ q)
    #     self.dec -= (q @ error.T)



class MLEI:
    def __init__(self, dt:float, T:float, NE:int=10, NI:int=10, I_e:float=90, I_i:float=90,
                 C:float=20, g_L:float=2, g_K:float=8, g_Ca:float=4.4, 
                 E_L:float=-60, E_K:float=-84, E_Ca:float=120, 
                 v1:float=-1.2, v2:float=18, v3:float=2, v4:float=30, phi:float=0.04,
                 a_r:float=1.1, a_d:float=0.19, v_t:float=2, k_p:float=5, t_max:float=1.0,
                 E_AMPA:float=0, E_GABA:float=-75, gbar:float=1) -> None:
        # Morris Lecar model parameters
        self._N = NE + NI
        self._NE = NE
        self._NI = NI
        self._I_e = I_e
        self._I_i = I_i
        self._C = C
        self._g_L = g_L
        self._g_K = g_K
        self._g_Ca = g_Ca
        self._E_L = E_L
        self._E_K = E_K
        self._E_Ca = E_Ca
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3
        self.v4 = v4
        self._phi = phi
        self._v_t = v_t
        self._a_r = a_r
        self._a_d = a_d
        self._k_p = k_p
        self._t_max = t_max
        self._E_AMPA = E_AMPA
        self._E_GABA = E_GABA
        self.gbar = gbar
        
        # time series
        self.ve = -50 + 30 * np.random.uniform(size=(NE, 1))
        self.se = np.random.rand(NE, 1)
        self.ne = np.zeros(shape=(NE, 1), dtype=float)
        self.vi = -50 + 30 * np.random.uniform(size=(NI, 1))
        self.si = np.random.rand(NI, 1)
        self.ni = np.zeros(shape=(NI, 1), dtype=float)
        
        # Network connections
        self.wee = np.random.rand(NE, NE) < 0.5
        self.wie = np.random.rand(NI, NE) < 0.5
        self.wei = np.random.rand(NE, NI) < 0.5
        self.wii = np.random.rand(NI, NI) < 0.5
        
        # Time series
        self._dt = dt
        self._duration = T
        self.time = np.arange(0, T, dt)
        self._nt = self.time.size

        # Post-Synaptic Potential
        self.isyne = np.zeros(shape=(NE, 1), dtype=float)
        self.isyni = np.zeros(shape=(NI, 1), dtype=float)
    
    # Excitatory functions
    def me_ss(self):
        return .5 * (1 + np.tanh((self.ve - self.v1) / self.v2))

    def ne_ss(self):
        return .5 * (1 + np.tanh((self.ve - self.v3) / self.v4))
    
    def tau_ne(self):
        return 1 / np.cosh((self.ve - self.v3) / (2 * self.v4))
    
    def T_e(self):
        return self._t_max / (1 + np.exp(-(self.ve - self._v_t) / self._k_p))
    
    def ne_dot(self):
        return self._phi * (self.ne_ss() - self.ne) / self.tau_ne()
    
    def se_dot(self):
        return self._a_r * self.T_e() * (1 - self.se) - self._a_d * self.se

    def ve_dot(self):
        self.isyne = self.gbar * (self.wee @ self.se) * (self.ve - self._E_AMPA) +\
            self.gbar * (self.wei @ self.si) * (self.ve - self._E_GABA)
        I_L = self._g_L * (self.ve - self._E_L)
        I_K = self._g_K * self.ne * (self.ve - self._E_K)
        I_Ca = self._g_Ca * self.me_ss() * (self.ve - self._g_Ca)
        return (self._I_e - I_L - I_K - I_Ca - self.isyne) / self._C

    # Excitatory functions
    def mi_ss(self):
        return .5 * (1 + np.tanh((self.vi - self.v1) / self.v2))

    def ni_ss(self):
        return .5 * (1 + np.tanh((self.vi - self.v3) / self.v4))
    
    def tau_ni(self):
        return 1 / np.cosh((self.vi - self.v3) / (2 * self.v4))
    
    def T_i(self):
        return self._t_max / (1 + np.exp(-(self.vi - self._v_t) / self._k_p))

    def ni_dot(self):
        return self._phi * (self.ni_ss() - self.ni) / self.tau_ni()
    
    def si_dot(self):
        return self._a_r * self.T_i() * (1 - self.si) - self._a_d * self.si

    def vi_dot(self):
        self.isyni = self.gbar * (self.wie @ self.se) * (self.vi - self._E_AMPA) +\
            self.gbar * (self.wii @ self.si) * (self.vi - self._E_GABA)
        I_L = self._g_L * (self.vi - self._E_L)
        I_K = self._g_K * self.ni * (self.vi - self._E_K)
        I_Ca = self._g_Ca * self.mi_ss() * (self.vi - self._g_Ca)
        return (self._I_i - I_L - I_K - I_Ca - self.isyni) / self._C

    def euler_step(self):
        dve = self._dt * self.ve_dot()
        self.ne += self._dt * self.ne_dot()
        self.se += self._dt * self.se_dot()
        self.ve += dve

        dvi = self._dt * self.vi_dot()
        self.ni += self._dt * self.ni_dot()
        self.si += self._dt * self.si_dot()
        self.vi += dvi

    def render(self):
        voltage_trace = np.zeros((self.time.size, self._N), dtype=float)

        for i in tqdm(range(self.time.size)):
            try:
                self.euler_step()
                voltage_trace[i] = np.concatenate([self.ve, self.vi], axis=1).ravel()
            except Exception as e:
                print(f"i = {i}")
                print(e)
                break

        return voltage_trace
    



def z_transform(signal):
    return (signal - signal.mean(axis=0)) / signal.std(axis=0)

def test():
    """Test the module"""
    T = 90
    dt = 1e-2
    t = np.arange(0, T, dt)
    nt = t.size
    x = np.sin(2 * .2 * np.pi * t / 1000)
    x = x.reshape(-1, 1)

    signal = z_transform(x)
    print(signal.shape)

    NE = 1
    NI = 1
    N = NI + NE
    
    # input current for I and E neurons
    Ie = 60
    Ii = 80
    current = np.ones((N, 1))
    middle = N // 2
    current[:middle] *= Ie  # NE bias
    current[middle:] *= Ii  # NI bias

    # RLS params
    rls_start = round(T * .05)
    rls_stop = round(T * .7)

    model = MorrisLecar(supervisor=signal, BIAS=current, T=T, dt=dt, 
                             N=N, Q=100, l=2.0)
    random_neurons, voltage_trace, _ = model.render(rls_start=rls_start, 
                                                 rls_stop=rls_stop, rls_step=50,
                                                 n_neurons=1)

    fig, ax = plt.subplots()
    for i in range(len(random_neurons)):
        signal = voltage_trace[:, i]
        minim = np.min(signal)
        maxim = np.max(signal)
        signal = (signal - minim) / (maxim - minim) + i
        ax.plot(t, signal)
    plt.grid(alpha=.5)
    # plt.savefig("img/test_voltage_trace.jpg", bbox_inches='tight')
    plt.show()

    plt.plot(t, x.ravel(), 'b', label="supervisor")
    plt.plot(t, model.x_hat_rec.ravel(), 'g', label="decoded")
    plt.grid(alpha=.5)
    plt.axvline(x=rls_start, c='r', label="start RLS")
    plt.axvline(x=rls_stop, c='cyan', label="stop RLS")
    plt.legend(loc=0)
    plt.savefig("img/rls_output.jpg", bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    test()