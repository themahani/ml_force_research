#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

from ml_block import MorrisLecarBlock, z_transform

def rmse(x, x_hat):
    """Return the Root Mean Squared Error for `x` vs `x_hat`.
    """
    return np.sqrt(np.mean((x - x_hat) ** 2))


def main():
    np.random.seed(1)
    
    Q_range = np.arange(1, 52, 10)
    lamda = 30.
    gbar_range = np.linspace(0., 0.05, 6)
    
    ### Global params for the model
    T = 20000
    dt = 1e-2
    t = np.arange(0, T, dt)
    nt = t.size
    x = np.sin(2 * .8 * np.pi * t / 1000)
    # y = np.cos(2 * .2 * np.pi * t / 1000)
    # x = np.vstack([x, y])
    # x = x.T
    x = x.reshape(-1, 1)
    # print(x.shape)
    signal = z_transform(x)
    print(signal.shape)

    NE = 100
    NI = 100
    N = NI + NE
    
    # input current for I and E neurons
    Ie = 95
    Ii = 95
    current = np.ones((N, 1))
    middle = N // 2
    current[:middle] *= Ie  # NE bias
    current[middle:] *= Ii  # NI bias
    # current = np.random.rand(N, 1) * Ie

    # RLS params
    rls_start = round(T * .02)
    rls_start = 1500
    rls_stop = round(T * .7)
    rls_step = 2

    for gbar in gbar_range:
        for Q in Q_range:            
            print(f"Running iteration for Q={Q}/{Q_range[-1]}, g={gbar}/{gbar_range[-1]}:\n")
            
            model = MorrisLecarBlock(supervisor=signal, BIAS=current, T=T, dt=dt, 
                                    N=N, Q=Q, l=lamda, gbar=gbar)

            random_neurons, voltage_trace = model.render(rls_start=rls_start, 
                                                        rls_stop=rls_stop, rls_step=rls_step,
                                                        live_plot=False, plt_interval=300,
                                                        n_neurons=10, save_all=False)
            
            # Get the train and test scores
            train_start = int(rls_start // dt)
            train_stop = int(rls_stop // dt)
            train_score = round(rmse(model.sup[train_start:train_stop], 
                                model.x_hat_rec[train_start:train_stop]), ndigits=5)                
            test_score = round(rmse(model.sup[train_stop:], 
                                model.x_hat_rec[train_stop:]), ndigits=5)
            
            # Plot the supervisor and the decoder signals
            fig, ax = plt.subplots(figsize=(10, 6), nrows=1, ncols=1)
            plt.plot(t, model.sup, 'b', label="supervisor")
            plt.plot(t, model.x_hat_rec, 'g', label="decoded")
            plt.ylim(-2, 2)
            plt.grid(alpha=.5)
            plt.axvline(x=rls_start, c='r', label="start RLS")
            plt.axvline(x=rls_stop, c='cyan', label="stop RLS")
            plt.legend(loc=0)
            plt.title(f"Q={Q}, l={lamda}, g={gbar}, T={T}, train={train_score}, test={test_score}")
            plt.savefig(f"tuning_results/train_{train_score}_test_{test_score}_Q_{Q}_l_{lamda}_gbar_{gbar}_rls_results.jpg",
                        bbox_inches='tight', dpi=250)
            plt.close()


if __name__ == "__main__":
    main()