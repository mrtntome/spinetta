
__all__ = ['PHundSys']

import numpy as np
import spinsys as sps
import lattice as ltc

class PHundSys:
    '''
    Class for Periodic Electron System with Hund Coupling on top of SpinSys
    '''
    def __init__(self, spincell):
        #if not (spincell.spintype is 'heisenberg'):
        #    raise ValueError('"spincell" must be of spintype "heisenberg".')
        self._spincell = spincell
        self._lattice = spincell._lattice
    
    def H(self, t_vec, J, k):
        #checks
        #extend for Lattice WB
        #if not isinstance(self._lattice, ltc.Lattice):
        #    raise ValueError('Only Bravais Lattice Supported for the moment.')
        num_sites = self._lattice._num_sites
        cells = self._lattice._cells
        prim_vecs = self._lattice._prim_vecs
        space_dim = self._lattice._space_dim
        neigbs = self._lattice._neigbs
        num_neigbs = self._lattice._num_neigbs
        neigbs_vecs = self._lattice._neigbs_vecs
        lb_S = self._spincell._lb_S
        H = np.zeros((2*num_sites, 2*num_sites), np.complex128)
        for i in range(num_sites):
            H[2*i, 2*i] = 0.5*J*lb_S[i, 2]
            H[2*i, 2*i+1] = 0.5*J*(lb_S[i, 0]-1j*lb_S[i, 1])
            H[2*i+1, 2*i] = 0.5*J*(lb_S[i, 0]+1j*lb_S[i, 1])
            H[2*i+1, 2*i+1] = -0.5*J*lb_S[i, 2]
            
            for n in range(num_neigbs):
                j = neigbs[i, n]
                if (j == -1):
                    continue
                t = t_vec[n] 
                delta = neigbs_vecs[i, n, :]
                
                H[2*i, 2*j] = -t * np.exp(1j*np.dot(k, delta))
                H[2*i+1, 2*j+1] = -t * np.exp(1j*np.dot(k, delta))
                H[2*j, 2*i] = -np.conj(t) * np.exp(-1j*np.dot(k, delta))
                H[2*j+1, 2*i+1] = -np.conj(t) * np.exp(-1j*np.dot(k, delta))
        return H

    def H_b(self, t, J, k):
        #checks
        #extend for Lattice WB
        #if not isinstance(self._lattice, ltc.Lattice):
        #    raise ValueError('Only Bravais Lattice Supported for the moment.')
        num_sites = self._lattice._num_sites
        cells = self._lattice._cells
        prim_vecs = self._lattice._prim_vecs
        space_dim = self._lattice._space_dim
        neigbs = self._lattice._neigbs
        num_neigbs = np.max(self._lattice._num_neigbs)
        neigbs_vecs = self._lattice._neigbs_vecs
        lb_S = self._spincell._lb_S
        H = np.zeros((2*num_sites, 2*num_sites), np.complex128)
        for i in range(num_sites):
            H[2*i, 2*i] = 0.5*J*lb_S[i, 2]
            H[2*i, 2*i+1] = 0.5*J*(lb_S[i, 0]-1j*lb_S[i, 1])
            H[2*i+1, 2*i] = 0.5*J*(lb_S[i, 0]+1j*lb_S[i, 1])
            H[2*i+1, 2*i+1] = -0.5*J*lb_S[i, 2]
            
            for n in range(np.max(num_neigbs)):
                j = neigbs[i, n]
                if (j == -1):
                    continue
                delta = neigbs_vecs[i, n, :]
                
                H[2*i, 2*j] = -t * np.exp(1j*np.dot(k, delta))
                H[2*i+1, 2*j+1] = -t * np.exp(1j*np.dot(k, delta))
                H[2*j, 2*i] = -np.conj(t) * np.exp(-1j*np.dot(k, delta))
                H[2*j+1, 2*i+1] = -np.conj(t) * np.exp(-1j*np.dot(k, delta))
        return H
    
    def V(self, xyz, t, k):
        num_sites = self._lattice._num_sites
        cells = self._lattice._cells
        prim_vecs = self._lattice._prim_vecs
        space_dim = self._lattice._space_dim
        neigbs = self._lattice._neigbs
        num_neigbs = self._lattice._num_neigbs
        neigbs_vecs = self._lattice._neigbs_vecs
        V = np.zeros((2*num_sites, 2*num_sites), np.complex128)
        
        for i in range(num_sites):
            V[2*i, 2*i] = -1j*t
            V[2*i+1, 2*i+1] = -1j*t
            
            for n in range(num_neigbs):
                j = neigbs[i, n]
                if (j == -1):
                    continue
                delta = neigbs_vecs[i, n, :]
                
                V[2*i, 2*j] = -1j*t * delta[xyz] * np.exp(-1j*np.dot(k, delta))
                V[2*i+1, 2*j+1] = -1j*t * delta[xyz] * np.exp(-1j*np.dot(k, delta))
                V[2*j, 2*i] =  1j*t * delta[xyz] * np.exp(1j*np.dot(k, delta))
                V[2*j+1, 2*i+1] = 1j*t * delta[xyz] * np.exp(1j*np.dot(k, delta))
        return V