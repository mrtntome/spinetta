
__all__ = ['SpinSys']

import numpy as np
import lattice as ltc

class SpinSys:
    '''
    Spin System Class
    '''
    def __init__(self, lattice, spintype='heisenberg', disp='rand'):
        check_lattice(lattice)
        check_built(lattice)

        n_sites = lattice.get_n_sites()

        #self._lattice = lattice
        self._S = gen_S(n_sites, spintype, disp)
        self.spintype = spintype
        self.built = False

    def get_S(self):
        return self._S

    def build(self, o1_int_array=None, o2_int_array=None):
         return 0

def check_lattice(lattice):
    if not (isinstance(lattice, ltc.LatticeWB) or isinstance(lattice, ltc.Lattice)):
        raise ValueError('"lat" must be a Lattice Object.')

def check_built(lattice):
    if (lattice.is_built() != True):
        raise ValueError('"lattice" isn\'t built.')

def gen_S(n_sites, spintype, disp):
    S = []
    if (spintype == 'heisenberg'):
        S = np.empty((n_sites, 3), dtype=np.float64)
        if (disp == 'rand'):
            phi = np.random.uniform(0.0, 2*np.pi, n_sites)
            theta = np.random.uniform(0.0, np.pi, n_sites)
            S[:, 0] = np.sin(theta) * np.cos(phi)
            S[:, 1] = np.sin(theta) * np.sin(phi)
            S[:, 2] = np.cos(theta)
    elif (spintype == 'XY'):
        S = np.empty((n_sites, 2), dtype=np.float64)
        if (disp == 'rand'):
            phi = np.random.uniform(0.0, 2*np.pi, n_sites)
            S[:, 0] = np.cos(phi)
            S[:, 1] = np.sin(phi)
    elif (spintype == 'ising'):
        S = np.empty((n_sites), dtype=np.int8)
        if (disp == 'rand'):
            S[:] = np.random.choice(np.array([-1, 1], dtype=np.int8), n_sites)
    else:
        raise ValueError('"spintype" must be one from: "heisenberg", "XY", "ising".')
    return S

class SpinCell:
    '''
    Spin Cell for spin superlattice
    '''
    def __init__(self, spinsys, sites, prim_vecs):
        S = spinsys.get_S()
        self.S = S[sites, :]
        self.prim_vecs = prim_vecs
        self.nsites = len(sites)

class spin_cell:
    '''
    Clase rápida
    '''
    def __init__(self, spins, lattice):
        if spins.shape[0] != lattice._num_sites:
            raise ValueError(' no coinciden los sitios')
        self._lb_S = spins
        self._lattice = lattice