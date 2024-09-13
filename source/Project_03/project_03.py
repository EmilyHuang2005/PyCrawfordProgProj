import numpy as np
from pyscf import gto
from scipy import linalg

class Molecule:

    def __init__(self):
        """charges: charges of atoms in molecule
        n: number of atoms in molecule
        coords: coordinates of atoms
        mol: pyscf molecule instance
        nao: number of atomic orbitals
        charge: charge of molecule
        nocc: number of occupied orbitals"""
        self.charges = NotImplemented
        self.n = NotImplemented
        self.coords = NotImplemented
        self.mol = NotImplemented
        self.path_dict = NotImplemented
        self.nao = NotImplemented
        self.charge = 0
        self.nocc = NotImplemented

    def construct_from_dat(self, filename: str):
        """Construct molecule from dat file"""
        with open(filename, 'r') as f:
            dat = np.array([line.split() for line in f.readlines()][1:], dtype=float)
            self.n = (np.shape(dat)[0])
            self.charges = np.array(dat[:, 0], dtype=int)
            self.coords = np.array(dat[:,1:], dtype = float)

    def obtain_mol_instance(self, basis = str, verbose=0):
        """Construct pyscf molecule instance
        input: basis set (str), verbose (int)"""
        mol = gto.Mole()
        mol.unit = 'Bohr'
        mol.charge = self.charge
        mol.atom = "".join(['{} {:25.20f} {:25.20f} {:25.20f}\n'.format(chg, *coord) for chg, coord in zip(self.charges, self.coords)])
        mol.basis = basis
        mol.charge = self.charge
        mol.spin = 0
        mol.verbose = verbose
        self.mol = mol.build()

    def eng_nuc_repulsion(self):
        """Calculate nuclear repulsion energy
        output: nuclear repulsion energy (float)"""
        return self.mol.energy_nuc()
    
    def obtain_nao(self):
        """Obtain number of atomic orbitals, store"""
        self.nao = self.mol.nao_nr()

    def integral_ovlp(self):
        """Calculate overlap integral
        output: overlap integral (np.ndarray, shape (nao, nao))"""
        return self.mol.intor("int1e_ovlp")

    def integral_kin(self):
        """Calculate kinetic integral
        output: kinetic integral (np.ndarray, shape (nao, nao))"""
        return self.mol.intor("int1e_kin")
    
    def integral_nuc(self):
        """Calculate nuclear integral
        output: nuclear integral (np.ndarray, shape (nao, nao))"""
        return self.mol.intor("int1e_nuc")

    def get_hcore(self):
        """Calculate core Hamiltonian
        output: Hamiltonian Core (np.ndarray, shape (nao, nao))"""
        # Output: Hamiltonian Core
        return self.integral_kin() + self.integral_nuc()
    
    def integral_eri(self):
        """Calculate electron repulsion integral
        output: electron repulsion integral (np.ndarray, shape (nao, nao, nao, nao))"""
        return self.mol.intor("int2e")
    
    def integral_ovlp_m1d2(self):
        """Build symmetric orthogonalization matrix
        output: symmetric orthogonalization matrix (np.ndarray, shape (nao, nao))"""
        return linalg.fractional_matrix_power(self.integral_ovlp(), -1/2)
    
    def get_fock(self, dm: np.ndarray):
        """Calculate Fock matrix
        input: density matrix (np.ndarray, shape (nao, nao))
        output: Fock matrix (np.ndarray, shape (nao, nao))"""
        return self.get_hcore() + np.einsum('kl, mnkl -> mn', dm, self.integral_eri()) - 0.5 * np.einsum('kl, mknl -> mn', dm, self.integral_eri())
    
    def get_coeff_from_fock(self, fock: np.ndarray):
        """Diagonalize Fock matrix
        input: Fock matrix (np.ndarray, shape (nao, nao))
        output: coefficient matrix (np.ndarray, shape (nao, nao))"""
        return linalg.eigh(fock, self.integral_ovlp())[1]
    
    def obtain_nocc(self):
        """Obtain number of occupied orbitals, store"""
        assert (np.sum(self.charges) - self.charge) % 2 == 0
        self.nocc = (np.sum(self.charges) - self.charge) // 2
        
    def make_rdm1(self, coeff: np.ndarray):
        """Build restricted density matrix
        input: coefficient matrix (np.ndarray, shape (nao, nao))
        output: restricted density matrix (np.ndarray, shape (nao, nao))"""
        return 2 * coeff[:, :self.nocc] @ coeff[:, :self.nocc].T
    
    def get_updated_dm(self, dm: np.ndarray):
        """Update density matrix
        input: density matrix (np.ndarray, shape (nao, nao))
        output: updated density matrix (np.ndarray, shape (nao, nao))"""
        fock = self.get_fock(dm)
        coeff = self.get_coeff_from_fock(fock)
        return self.make_rdm1(coeff)
    
    def eng_total(self, dm: np.ndarray):
        """Calculate total energy
        input: density matrix (np.ndarray, shape (nao, nao))
        output: total energy in Hartrees(float)"""
        return 1/2 * (np.sum(dm * (self.get_hcore() + self.get_fock(dm)))) + self.eng_nuc_repulsion()

    def scf_process(self, dm_guess: np.ndarray=None):
        """Perform SCF process
        input: initial density matrix (np.ndarray, shape (nao, nao))
        output: total energy (float), density matrix (np.ndarray, shape (nao, nao))"""
        eng, dm = 0., np.zeros((self.nao, self.nao)) if dm_guess is None else np.copy(dm_guess)
        eng_next, dm_next = NotImplemented, NotImplemented
        max_iter, thresh_eng, thresh_dm = 64, 1e-10, 1e-8
        print("{:>5} {:>20} {:>20} {:>20}".format("Epoch", "Total Energy", "Energy Deviation", "Density Deviation"))
        for epoch in range(max_iter):
            eng_next = self.eng_total(dm)
            dm_next = self.get_updated_dm(dm)
            print("{:5d} {:20.12f} {:20.12f} {:20.12f}".format(epoch, eng_next, eng_next - eng, np.linalg.norm(dm_next - dm)))
            if np.abs(eng_next - eng) < thresh_eng and np.linalg.norm(dm_next - dm) < thresh_dm:
                break
            eng, dm = eng_next, dm_next
        return eng, dm
    
    def integral_dipole(self):
        """Calculate dipole related integral
        output: dipole related integral (np.ndarray, shape (3, nao, nao))"""
        return self.mol.intor("int1e_r")
    
    def get_dipole(self, dm: np.ndarray):
        """Calculate dipole moment
        input: density matrix (np.ndarray, shape (nao, nao))
        output: dipole moment (np.ndarray, shape (3))"""
        return -1 * (np.sum(dm * mlc.integral_dipole(), axis=(2, 1))) + np.sum(self.coords * self.charges[:, None], axis = 0)

    def population_analysis(self, dm: np.ndarray):
        """Perform Mulliken population analysis
        input: density matrix (np.ndarray, shape (nao, nao))
        output: Mulliken population (np.ndarray, shape (nao))"""
        pop = np.zeros(self.n)
        dens_ovlp = dm @ self.integral_ovlp()
        ao_indices = self.mol.aoslice_by_atom()
        for i in range(self.n):
            _, _, i0, i1 = ao_indices[i]
            pop[i] = self.charges[i] - np.trace(dens_ovlp[i0:i1, i0:i1])
        return pop

if '__main__' == __name__:
    mlc = Molecule()
    mlc.construct_from_dat('input/h2o/STO-3G/geom.dat')
    mlc.obtain_mol_instance('STO-3G')
    mlc.obtain_nao()
    np.random.seed(0)
    dm_random = np.random.randn(mlc.nao, mlc.nao)
    dm_random += dm_random.T
    mlc.obtain_nocc()
    dm_guess = np.zeros((mlc.nao, mlc.nao)) 
    eng, dm_converged = mlc.scf_process(dm_guess)
    print(mlc.get_dipole(dm_converged))
    print(mlc.population_analysis(dm_converged))

    


    

