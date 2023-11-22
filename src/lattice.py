
__all__ = ['Lattice', 'LatticeWB']

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections  as mc
import itertools

class Lattice:
    '''
    Bravais Lattice Class
    '''
    def __init__(self, prim_vecs, offset=None):
        prim_vecs = np.array(prim_vecs, dtype=np.float64)
        check_prim_vecs(prim_vecs)

        space_dim = prim_vecs.shape[0]

        if (offset is None):
            offset = np.zeros(space_dim)
        else:
            offset = np.array(offset, dtype=np.float32)
            check_offset(offset, space_dim)

        self.sublattices = [self]
        self._prim_vecs = prim_vecs
        self._space_dim = space_dim
        self._offset = offset
        self._built = False

    def get_prim_vecs(self):
        return self._prim_vecs

    def get_space_dim(self):
        return self._space_dim

    def get_offset(self):
        return self._offset

    def is_built(self):
        return self._built

    def get_num_sites(self):
        if (self._built is True):
            return self._num_sites
        else:
            raise ValueError('Lattice isn\'t built.')

    def get_coords(self, site):
        if self._built is False:
            raise ValueError('"lattice" isnt built.')
        # check site
        return self._coords[tuple(site)]
    
    def get_label(self, site):
        if self._built is False:
            raise ValueError('"lattice" isnt built.')
        # check site
        return self._labels[tuple(site)]
    
    def get_site(self, label):
        if self._built is False:
            raise ValueError('"lattice" isnt built.')
        # check site
        return self._sites[label]
    
    def get_neigbs(self, label):
        neigbs = self._neigbs[label, :]
        neigbs = np.extract(neigbs != -1, neigbs)
        return neigbs

    def plot(self):
        space_dim = self._space_dim
        num_sites = self._num_sites
        coords = self._coords
        serial_coords = np.reshape(coords, (num_sites, space_dim), 'C')
        if (space_dim != 2):
            raise ValueError('Only dimension 2 supported for the moment')
        ax = plt.subplot()

        ax.scatter(serial_coords[:, 0], serial_coords[:, 1])
        if hasattr(self, '_neigbs_vecs'):
            neigbs_vecs = self._neigbs_vecs
            lines = []
            for i in range(num_sites):
                x0 = serial_coords[i]
                for j in range(neigbs_vecs.shape[1]):
                    if not np.any(neigbs_vecs[i, j, :]):
                        continue
                    xf = x0 + neigbs_vecs[i, j, :]
                    lines.append([x0, xf])
            lines = mc.LineCollection(lines)
            ax.add_collection(lines)
            
        plt.show()

    def build(self, cells, boundary=[], delta_vecs=[]):
        space_dim = self._space_dim
        prim_vecs = self._prim_vecs
        offset = self._offset
        num_cells = np.prod(cells)
        num_sites = num_cells
        coords = np.zeros(np.append(cells, space_dim)) + offset
        labels = np.zeros(cells, dtype=np.int64)
        sites = np.zeros([num_sites, space_dim], dtype=np.int32)
        #check_n_cells(n_cells, space_dim) 
        #check nbgs

        for site in np.ndindex(tuple(cells)):
            for d in range(space_dim):
                coords[tuple(site)] += site[d] * prim_vecs[d, :]
        
        site_iter = np.ndindex(tuple(cells))
        for i in range(num_sites):
            site = site_iter.next()
            labels[tuple(site)] = i
            sites[i, :] = site
        
        if (boundary != []):
            num_neigbs = delta_vecs.shape[0]
            neigbs = np.full((num_sites, num_neigbs), -1, dtype=np.int32)
            neigbs_vecs = np.zeros((num_sites, num_neigbs, space_dim), dtype=np.float64)
            for i in range(num_sites):
                site1 = sites[i]
                site2 = np.full(space_dim, -1, dtype=np.int64)
                for j in range(num_neigbs):
                    delta = delta_vecs[j, :] 
                    for d in range(space_dim):
                        if (boundary[d] == 'p'):
                            site2[d] = np.remainder(site1[d] + delta[d], cells[d])
                        elif (boundary[d] == 'o'):
                            site2[d] = site1[d] + delta[d]
                            if ((site2[d] < 0) or (site2[d] >= cells[d])):
                                site2[d] = -1
                                break
                        else:
                            raise ValueError('Only supported BC for the moment: periodic="p" , open="o".')
                    if np.in1d(-1, site2):
                        continue
                    else:
                        neigbs[i, j] = labels[tuple(site2)]
                        for d in range(space_dim):
                            neigbs_vecs[i, j, :] += delta[d] * prim_vecs[d, :]
            self._boundary = boundary
            self._delta_vecs = delta_vecs
            self._num_neigbs = num_neigbs
            self._neigbs = neigbs
            self._neigbs_vecs = neigbs_vecs
                
        self._cells = cells
        self._num_cells = num_cells 
        self._num_sites = num_sites
        self._coords = coords
        self._labels = labels
        self._sites = sites
        self._built = True

class LatticeWB(Lattice):
    '''
    Lattice with Basis Class
    '''
    def __init__(self, prim_vecs, basis_vecs=None, offset=None):
        prim_vecs = np.array(prim_vecs, dtype=np.float32)
        check_prim_vecs(prim_vecs)

        space_dim = prim_vecs.shape[0]

        if (basis_vecs is None):
            basis_vecs = np.reshape(np.zeros(space_dim, dtype=np.float32), (1, 2))
        else:
            basis_vecs = np.array(basis_vecs, dtype=np.float32)
            check_basis(basis_vecs, space_dim)

        if (offset is None):
            offset = np.zeros(space_dim)
        else:
            offset = np.array(offset, dtype=np.float32)
            check_offset(offset, space_dim)

        num_basis_vecs = basis_vecs.shape[0]
        sublattices = [Lattice(prim_vecs, vector + offset) for vector in basis_vecs]
        num_sublattices = len(sublattices)
        
        self._prim_vecs = prim_vecs
        self._basis_vecs = basis_vecs
        self._offset = offset
        self._space_dim = space_dim
        self._num_basis_vecs = num_basis_vecs
        self._sublattices = sublattices
        self._num_sublattices = num_sublattices
        self._built = False

    def get_basis_vecs(self):
        return self._basis_vecs

    def build(self, cells, boundary=[], *delta_vecs):
        space_dim = self._space_dim
        prim_vecs = self._prim_vecs
        basis_vecs = self._basis_vecs
        num_basis_vecs = self._num_basis_vecs
        sublattices = self._sublattices
        num_sublattices = self._num_sublattices
        num_cells = np.prod(cells)
        num_sites = num_cells * num_basis_vecs
        coords = np.zeros(np.append(cells, [num_basis_vecs, space_dim]))
        labels = np.zeros(np.append(cells, num_basis_vecs), dtype=np.int64)
        sites = np.zeros([num_sites, space_dim + 1], dtype=np.int32)

        for lattice in sublattices:
            lattice.build(cells)

        for i in range(num_sublattices):
            coords[..., i, :] = sublattices[i]._coords

        site_iter = np.ndindex(tuple(np.append(cells, num_basis_vecs)))
        for i in range(num_sites):
            site = site_iter.next()
            labels[tuple(site)] = i
            sites[i, :] = site
            
        if (boundary != []):
            num_neigbs = [dv.shape[0] for dv in delta_vecs] 
            neigbs = np.full((num_sites, np.max(num_neigbs)), -1, dtype=np.int32)
            neigbs_vecs = np.zeros((num_sites, np.max(num_neigbs), space_dim), dtype=np.float64)
            for i in range(num_cells):
                for b in range(num_basis_vecs):
                    site1 = sites[i*num_basis_vecs + b]
                    site2 = np.full(space_dim + 1, -1, dtype=np.int64)
                    nn = num_neigbs[b]
                    dv = delta_vecs[b]
                    if (nn == 0):
                        continue
                    for j in range(nn):
                        delta = dv[j, :] 
                        site2[-1] = site1[-1] + delta[-1] 
                        if ((site2[-1] < 0) or (site2[-1] >= num_basis_vecs)):
                            continue
                        for d in range(space_dim):
                            if (boundary[d] == 'p'):
                                site2[d] = np.remainder(site1[d] + delta[d], cells[d])
                            elif (boundary[d] == 'o'):
                                site2[d] = site1[d] + delta[d]
                                if ((site2[d] < 0) or (site2[d] >= cells[d])):
                                    site2[d] = -1
                                    break
                            else:
                                raise ValueError('Only supported BC for the moment: periodic="p" , open="o".')
                        if np.in1d(-1, site2):
                            continue
                        else:
                            neigbs[i*num_basis_vecs+b, j] = labels[tuple(site2)]
                            neigbs_vecs[i*num_basis_vecs+b, j, :] += basis_vecs[site2[-1], :] - basis_vecs[b, :]
                            for d in range(space_dim):
                                neigbs_vecs[i*num_basis_vecs+b, j, :] += delta[d] * prim_vecs[d, :]
            self._boundary = boundary
            self._delta_vecs = delta_vecs
            self._num_neigbs = num_neigbs
            self._neigbs = neigbs
            self._neigbs_vecs = neigbs_vecs
                
        self._cells = cells
        self._num_cells = num_cells 
        self._num_sites = num_sites
        self._coords = coords
        self._labels = labels
        self._sites = sites
        self._built = True


def check_prim_vecs(prim_vecs):
    """
    Checker for Primitive Bravais Vector
    """
    if (prim_vecs.ndim != 2):
        raise ValueError('"prim_vecs" must be a 2d array.')

    if (prim_vecs.shape[0] != prim_vecs.shape[1]):
        raise ValueError('Number of Primitive Vectors must be same as the space dimensionality.')

    if (np.linalg.matrix_rank(prim_vecs) != prim_vecs.shape[0]):
        raise ValueError('Primitive Vectors must be linearly independent')

def check_basis(basis, space_dim):
    if (basis.ndim != 2):
        raise ValueError('"basis" must be a 2d array.')
    if (basis.shape[1] != space_dim):
        raise ValueError('Basis dimensionality does not match Space Dimensionality.')

def check_offset(offset, space_dim):
    if (offset.ndim != 1):
        raise ValueError('"offset" must be a 1d array.')
    if (offset.shape[0] != space_dim):
        raise ValueError('"offset" dimensionality does not match Space Dimensionality')

## TODO
def check_n_cells():
    return 0
