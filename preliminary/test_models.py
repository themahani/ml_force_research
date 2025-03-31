"""This module includes various test models to use in testing the FORCE method.
"""

import numpy as np


class LorenzAttractor:
    def __init__(self, xyz:np.ndarray=None,
                 sigma:float=10, rho:float=28, beta:float=2.667) -> None:
        """The Lorenz attractor initial conditions and parameters.
        
        If initial condition is `None` type, will use random initial conditions.

        Parameters
        ----------
        xyz : np.ndarray, optional
            Initial condition of the simulation for x, y, and z, by default None
        sigma : float, optional
            Sigma parameter of the model, by default 10
        rho : float, optional
            rho parameter of the model, by default 28
        beta : float, optional
            beta parameter of the model, by default 2.667
        """
        self.b = beta
        self.s = sigma
        self.r = rho
        
        if xyz is None:
            self.xyz = np.random.rand(3, 1)
        else:
            if xyz.shape != (3, 1):
                raise ValueError(f"xyz should be of shape (3, 1) not {xyz.shape}")
            self.xyz = xyz
            
    def __xyz_dot(self) -> np.ndarray:
        """Calculate the time derivative of each variable and return it

        Returns
        -------
        np.ndarray
            An array of the time derivative of variables of the model with 
            shape (3, 1)
        """
        x, y, z = self.xyz.flatten()
        
        x_dot = self.s * (y - x)
        y_dot = self.r * x - y - x * z
        z_dot = x * y - self.b * z
        
        return np.array([[x_dot], [y_dot], [z_dot]])
    
    def render(self, T:float=2000, dt:float=4e-2) -> [np.ndarray, np.ndarray]:
        """_summary_

        Parameters
        ----------
        T : float, optional
        Duration of the simulation in [ms], by default 2000
        dt : float, optional
            Time step of the simulation in [ms], by default 4e-2

        Returns
        -------
        np.ndarray
            The time array of shape (nt, ) where `nt` is the number of time steps
        
        np.ndarray
            Recorded time series for `xyz` with shape (nt, `xyz.size`)
        """
        time = np.arange(0, T, dt)
        nt = time.size
        xyz_rec = np.zeros(shape=(nt, self.xyz.size), dtype=float)
        for i in range(nt):
            self.xyz += dt * self.__xyz_dot()
            xyz_rec[i] = self.xyz.flatten()
        
        return time, xyz_rec
    
        