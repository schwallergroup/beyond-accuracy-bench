#!/usr/bin/env python
# coding: utf-8

"""
This module provides a LogProb class for calculating log-likelihood using
the `basico` library for kinetic simulations.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from numba import jit
import inspect


# Ensure the project root is in the path
script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class LogProb:
    """A class to calculate the log probability for emcee, using scipy for simulation."""
    def __init__(self, multi_run_data: dict, param_names: list[str], 
                 ode_func, species_order: list[str],
                 prior_config: dict | None = None, fixed_sigma: float | None = None):
        """
        Initializes the LogProb calculator.
        """
        self.multi_run_data = multi_run_data
        self.fixed_sigma = fixed_sigma
        
        self.param_names = list(param_names) # Copy to avoid modifying the original list
        if self.fixed_sigma is not None:
            if 'log_sigma' in self.param_names:
                self.param_names.remove('log_sigma')

        self.n_params = len(self.param_names)
        self.param_map = {name: i for i, name in enumerate(self.param_names)}

        prior_config = prior_config or {}
        self.log_k_min = prior_config['log_k_min']
        self.log_k_max = prior_config['log_k_max']
        self.log_sigma_min = prior_config['log_sigma_min']
        self.log_sigma_max = prior_config['log_sigma_max']
        self.log_conc_min = prior_config['log_conc_min']
        self.log_conc_max = prior_config['log_conc_max']

        self.n_data = 0
        for run_data in self.multi_run_data.values():
            if 'substrate_data' in run_data:
                self.n_data += len(run_data['time_data'])
            if 'product_data' in run_data:
                self.n_data += len(run_data['time_data'])

        # Scipy ODE setup from passed arguments
        self.species_order = species_order
        self.ode_func = ode_func

    def log_prior(self, theta: np.ndarray) -> float:
        for name, value in zip(self.param_names, theta):
            if 'log_(' in name and '.k' in name:
                if not (self.log_k_min < value < self.log_k_max): return -np.inf
            elif 'log_sigma' in name:
                if not (self.log_sigma_min < value < self.log_sigma_max): return -np.inf
        return 0.0

    def log_likelihood(self, theta: np.ndarray) -> float:
        # Unpack sigma from the end of theta
        if self.fixed_sigma is None:
            log_sigma = theta[self.param_map['log_sigma']]
            sigma = 10**log_sigma
        else:
            sigma = self.fixed_sigma

        # Extract k_values directly from theta, preserving order, and convert to tuple for solve_ivp
        k_indices = [i for i, name in enumerate(self.param_names) if '.k' in name]
        k_values = 10**theta[k_indices]
        k_args = tuple(k_values)

        total_sse = 0.0

        for run_data in self.multi_run_data.values():
            t_exp = np.array(run_data['time_data'])
            
            # --- Prepare arguments for scipy's solve_ivp ---
            
            # 1. Initial conditions vector (y0)
            initial_conditions = run_data.get('initial_conditions', {})
            y0 = np.zeros(len(self.species_order))
            for i, species_name in enumerate(self.species_order):
                y0[i] = initial_conditions.get(species_name, 0.0)

            # 2. Kinetic parameters tuple (k_args) is now correctly prepared above

            # --- Solve ODE using scipy ---
            sol = solve_ivp(
                fun=self.ode_func,
                t_span=(t_exp[0], t_exp[-1]),
                y0=y0,
                t_eval=t_exp,
                args=k_args,
                method='LSODA',  # LSODA is good for stiff problems
                rtol=1e-4,
                atol=1e-8
            )
            
            if not sol.success:
                return -np.inf

            # Create a pandas DataFrame from the results for compatibility
            sol_df = pd.DataFrame(sol.y.T, index=sol.t, columns=self.species_order)

            # --- Calculate SSE for available data ---
            if 'substrate_data' in run_data and 'S' in self.species_order:
                y_obs_S = np.array(run_data['substrate_data'])
                sim_S = sol_df['S'].values
                total_sse += np.sum((sim_S - y_obs_S)**2)

            if 'product_data' in run_data and 'P' in self.species_order:
                y_obs_P = np.array(run_data['product_data'])
                sim_P = sol_df['P'].values
                total_sse += np.sum((sim_P - y_obs_P)**2)

        if self.n_data == 0:
            return -np.inf
        
        log_like = -0.5 * self.n_data * np.log(2 * np.pi * sigma**2) - 0.5 * total_sse / (sigma**2)
        return log_like

    def __call__(self, theta: np.ndarray) -> float:
        lp = self.log_prior(theta)
        if not np.isfinite(lp): return -np.inf
        ll = self.log_likelihood(theta)
        if not np.isfinite(ll): return -np.inf
        return ll