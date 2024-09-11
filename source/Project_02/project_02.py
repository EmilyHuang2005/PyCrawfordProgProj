import numpy as np
from molmass import ELEMENTS

class Molecule:

    def __init__(self):
        self.n = NotImplemented
        self.charges = NotImplemented
        self.coordinates = NotImplemented
        self.hess = NotImplemented

    def read_dat(self, filename: str):
        with open(filename, 'r') as f:
            dat = np.array([line.split() for line in f.readlines()][1:], dtype=float)
            self.n = (np.shape(dat)[0])
            self.charges = np.array(dat[:, 0], dtype=int)
            self.coordinates = np.array(dat[:,1:], dtype = float)

    def obtain_hessian(self, hess_file: str):
        with open(hess_file, 'r') as f:
            self.hess = np.array([line.split() for line in f.readlines()][1:], dtype=float).reshape(self.n * 3, self.n * 3)

    def mass_weighted_hessian(self):
        masses = np.repeat(np.array(1 / np.sqrt(ELEMENTS[c].mass for c in self.charges)), 3)
        return np.einsum('i,j,ij->ij', masses, masses, self.hess)

    def eig_mass_weighted_hessian(self):
        return np.linalg.eigvals(self.mass_weighted_hessian())
    
    
    
    