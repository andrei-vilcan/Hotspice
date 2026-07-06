from typing import Optional

import ASI
from .core import Scheme, SimParams

import numpy as np
import scipy.constants as const

from .energies import DiMonopolarEnergy, SensoryLayerEnergy

# Boltzmann's constant
k_B = const.k


class TrackingEnvironment:
    def __init__(self, V, alpha, x_0, dx_0):
        self.V = V
        self.alpha = alpha
        self.x = x_0
        self.dx = dx_0

    def step(self, a, dt=1.0):
        self.dx = -self.alpha * self.x + a
        self.x += self.dx * dt

    def get_sensory_input(self):
        return self.x, self.dx

    def reset(self):
        self.x = 0.0
        self.dx = 0.0


class SensoryLayer:
    """
    Ising spins

    each spin is +/- 1 along a fixed axis
    updated by Metropolis-Hastings Monte Carlo
        according to external applied field (environment)

    """
    def __init__(self, n: int, alignment: float, moment: float, n_mcs: int, rng: np.random.Generator):
        self.n = n
        self.alignment = alignment
        self.moment = moment
        self.n_mcs = n_mcs

        # t_n_m_o = (2 * n) - 1
        # self.sigma_sparse = 1 - np.indices((t_n_m_o, t_n_m_o)).sum(axis=0) % 2

        self.rng = rng
        self.sigma = self.rng.choice([-1.0, 1.0], size=n)

    def update(self, h_e, beta_s, n_mcs = 1):
        """
        energy calculated according to eq (23) of Stamps
        (with neglections under assumptions)

        in each Monte Carlo simulation attempt to flip each spin once
        """
        n_mcs = self.n_mcs if self.n_mcs is not None else n_mcs

        for _ in range(n_mcs):
            for i in range(self.n):
                dE = -2.0 * beta_s * self.moment * h_e * self.sigma[i]
                if self.rng.random() < np.exp(min(0.0, dE)):
                    self.sigma[i] *= -1

    def reset(self, pattern = 'random'):
        if pattern == 'random':
            self.sigma = self.rng.choice([-1, 1], size=self.n).astype(float)
        elif pattern == 'up':
            self.sigma = np.ones(self.n)
        elif pattern == 'down':
            self.sigma = -np.ones(self.n)


def create_hidden_layer(a, n, **kwargs):
    sim_params = SimParams(
        UPDATE_SCHEME=Scheme.METROPOLIS,
        MULTISAMPLING_SCHEME='single',
        ENERGY_BARRIER_METHOD='simple'
    )
    defaults = dict(
        pattern='random',
        params=sim_params,
    )
    defaults.update(kwargs)

    si = ASI.IP_Square_Open_Shifted(a, n, **defaults)
    return si


class BilayerASI:
    """
    Two-layer system with sensory and hidden layer
    """
    def __init__(
            self,
            n: int,
            a: float,
            alignment: float,
            m_s: float,
            m_h: float,
            N_s: int,
            N_h: int,
            beta_s: float,
            beta_h: float,

            rng: np.random.Generator):
        """
        :param n: number of cells per side
        :param a: lattice constant and side length
        :param alignment: alignment of the sensory spins
        :param m_s: magnetic moment of sensory spins
        :param m_h: magnetic moment of hidden spins
        :param N_s: number of Monte Carlo Steps performed during each time interval for sensory spins
        :param N_h: number of Monte Carlo Steps performed during each time interval for hidden spins
        :param beta_s: inverse temperature of sensory layer
        :param beta_h: inverse temperature of hidden layer

        :param rng: numpy generator
        """
        # params
        self.n = n

        n_sensors = lambda x: x**2 + (x - 1)**2
        self.n_sensors = n_sensors(n)

        n_hidden = lambda x: 4 * x**2
        self.n_hidden = n_hidden(n)

        self.a = a
        self.l = a / 2.0
        self.z_sep = a / 10.0

        self.alignment = alignment

        self.m_s = m_s
        self.m_h = m_h
        self.N_s = N_s
        self.N_h = N_h

        self.beta_s = beta_s
        self.beta_h = beta_h

        self.rng = rng

        # create layers
        self.sensory = SensoryLayer(self.n_sensors, self.alignment, self.m_s, self.N_s, rng=self.rng)
        self.hidden = create_hidden_layer(self.a, self.n)

        # define energy of each layer and add to hidden layer
        hidden_energy = DiMonopolarEnergy(d=self.l)
        sensory_energy = SensoryLayerEnergy(self.sensory)
        self.hidden.add_energy(hidden_energy)
        self.hidden.add_energy(sensory_energy)

    def update_sensory(self, x):
        self.sensory.update(h_e=x, beta_s=self.beta_s)

    def update_hidden(self,):
        self.hidden.progress(self.N_h)

    def update(self, x):
        self.update_sensory(x)
        self.update_hidden()

    def action(self):
        pass


class ActiveInferenceSimulation:
    def __init__(
            self,
            n: int,
            a: Optional[float] = None,
            alignment: Optional[float] = None,
            m_s: Optional[float] = None,
            m_h: Optional[float] = None,
            N_s: Optional[int] = None,
            N_h: Optional[int] = None,
            T_s: Optional[float] = None,
            T_h: Optional[float] = None,

            env: Optional[str] = None,
            env_params: Optional[dict] = None,

            seed: Optional[int] = None,
    ):
        """
        :param n: number of cells per side in bilayer ASI arrangement
        :param a: lattice constant and side length
        :param m_s: magnetic moment of sensory spins
        :param m_h: magnetic moment of hidden spins
        :param N_s: number of Monte Carlo Steps performed during each time interval for sensory spins
        :param N_h: number of Monte Carlo Steps performed during each time interval for hidden spins
        :param T_s: effective temperature of sensory layer
        :param T_h: effective temperature of hidden layer

        :param env: environment choice
        :param env_params: parameters for specific environment

        :param seed: numpy rng seed
        """

        # 'hardware'
        self.n = n

        # default system parameters
        self.a         = a         if a         is not None else 2e-9            # square side length          [m]
        self.alignment = alignment if alignment is not None else 0               # alignment of sensory spins
        self.m_s       = m_s       if m_s       is not None else 1.1278401e-15   # magnetic moment             [Am²]
        self.m_h       = m_h       if m_h       is not None else self.m_s * 0.2  # magnetic moment             [Am²]
        self.N_s       = N_s       if N_s       is not None else 10              # number of sensory layer Monte Carlo steps
        self.N_h       = N_h       if N_h       is not None else 1               # number of hidden layer Monte Carlo steps
        self.beta_s    = self.m_s / (k_B * T_s)                                  # sensory layer inverse temperature
        self.beta_h    = self.m_h / (k_B * T_h)                                  # hidden layer inverse temperature

        # environment selection
        if env == 'tracking':
            self.environment = TrackingEnvironment(**env_params)
        else:
            raise ValueError(f"unknown environment: {env!r}")

        # other params
        self._seed = seed
        self._rng = np.random.default_rng(seed)

        # create system
        self.system = self.create_bilayerASI()

    def create_bilayerASI(self, **kwargs):
        defaults = dict(
            n = self.n,
            a = self.a,
            alignment = self.alignment,
            m_s = self.m_s,
            m_h = self.m_h,
            N_s = self.N_s,
            N_h = self.N_h,
            beta_s = self.beta_s,
            beta_h = self.beta_h,
            _rng = self._rng
        )
        defaults.update(kwargs)
        return BilayerASI(**defaults)

    def run_simulation(self):
        pass
