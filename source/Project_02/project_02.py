import numpy as np
from molmass import ELEMENTS
from scipy import constants as const

centi = const.centi
c = const.c
E_h = const.physical_constants['Hartree energy'][0]
amu = const.physical_constants['atomic mass constant'][0]
a_0 = const.physical_constants['Bohr radius'][0]

class Molecule:

    def __init__(self):
        self.n = NotImplemented
        self.charges = NotImplemented
        self.coordinates = NotImplemented
        self.hess = NotImplemented

    def read_dat(self, filename: str):
        """Read the input file, stores the number of atoms, charges and coordinates"""
        with open(filename, 'r') as f:
            dat = np.array([line.split() for line in f.readlines()][1:], dtype=float)
            self.n = (np.shape(dat)[0])
            self.charges = np.array(dat[:, 0], dtype=int)
            self.coordinates = np.array(dat[:,1:], dtype = float)

    def obtain_hessian(self, hess_file: str):
        """Read the Hessian matrix from the input file, store as symmetric matrix"""
        with open(hess_file, 'r') as f:
            self.hess = np.array([line.split() for line in f.readlines()][1:], dtype=float).reshape(self.n * 3, self.n * 3)

    def mass_weighted_hessian(self):
        """Calculate the mass-weighted Hessian matrix
        output: mass-weighted Hessian matrix (np.ndarray)"""
        masses = np.repeat(np.array([1 / np.sqrt(ELEMENTS[c].mass) for c in self.charges]), 3)
        return np.einsum('i,j,ij->ij', masses, masses, self.hess)

    def eig_mass_weighted_hessian(self):
        """Calculate the Hermitian eigenvalues of the mass-weighted Hessian matrix
        output: Hermitian eigenvalues of mass-weighted Hessian matrix (np.ndarray)"""
        return np.linalg.eigvalsh(self.mass_weighted_hessian())
    
    def harmonic_vib_freq(self):
        """Calculate the harmonic vibrational frequencies
        output: harmonic vibrational frequencies (np.ndarray)"""
        eig = self.eig_mass_weighted_hessian()
        return np.sign(eig) * np.sqrt(np.abs(eig)) * centi/(2*np.pi*c) * (np.sqrt(E_h/amu/a_0**2))
    
    def print_solution_02(self):
        """Print the solution for Project 02"""
        print("=== Mass-Weighted Hessian Matrix (in unit Eh/(amu*a0^2)) ===")
        print(self.mass_weighted_hessian())
        print("=== Eigenvalues of Mass-Weighted Hessian Matrix (in unit Eh/(amu*a0^2)) ===")
        print(self.eig_mass_weighted_hessian())
        print("=== Harmonic Vibrational Frequencies (in unit cm^-1) ===")
        print(self.harmonic_vib_freq())

if __name__ == '__main__':
    mol = Molecule()
    mol.read_dat('input/benzene_geom.txt')
    mol.obtain_hessian('input/benzene_hessian.txt')
    mol.print_solution_02()