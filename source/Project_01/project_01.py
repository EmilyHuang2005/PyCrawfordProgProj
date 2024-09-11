import numpy as np
from molmass import ELEMENTS
import scipy.constants as const

a_0 = const.physical_constants['Bohr radius'][0] #Bohr radius in m
ang = const.angstrom #Angstrom in m
amu = const.physical_constants['atomic mass constant'][0]   #atomic mass constant in kg
kilo = const.kilo
centi = const.centi
mega = const.mega
h = const.h #Planck constant in J s
c = const.c #speed of light

class Molecule:

    def __init__(self):
        """Initialize Molecule object
        n: number of atoms
        charges: atomic charges
        coordinates: atomic coordinates"""
        self.n = NotImplemented
        self.charges = NotImplemented
        self.coordinates = NotImplemented

    def read_dat(self, filename: str):
        """Read the input file, stores the number of atoms, charges and coordinates"""
        with open(filename, 'r') as f:
            dat = np.array([line.split() for line in f.readlines()][1:], dtype=float)
            self.n = (np.shape(dat)[0])
            self.charges = np.array(dat[:, 0], dtype=int)
            self.coordinates = np.array(dat[:,1:], dtype = float)

    def bond_length(self, i: int, j: int) -> float:
        """Calculate the bond length between atoms i and j
        input: atomic indices i, j (int)
        output: bond length in Bohr (float)"""
        return np.linalg.norm(self.coordinates[i] - self.coordinates[j])
            
    def print_bond_lengths(self):
        """Print the bond lengths between all pairs of atoms"""
        print('===Bond Length===')
        for i in range(self.n):
            for j in range(i+1, self.n):
                print(f'{i:3d} - {j}: {self.bond_length(i, j):12.5f} Bohr')

    def bond_unit_vector(self, i: int, j: int) -> np.ndarray:
        """Calculate the unit vector pointing from atom i to atom j (np.ndarray)
        input: atomic indices i, j
        output: unit vector"""
        return (self.coordinates[j] - self.coordinates[i]) / np.linalg.norm(self.coordinates[j] - self.coordinates[i])  

    def bond_angle(self, i: int, j: int, k: int) -> float:
        """Calculate the bond angle between atoms i, j and k
        input: atomic indices i, j, k (int)
        output: bond angle in degrees (float)"""
        return np.degrees(np.arccos(self.bond_unit_vector(j, i).dot(self.bond_unit_vector(j, k))))

    def is_valid_angle(self, i: int, j: int, k: int) -> bool:
        """Check if the bond angle between atoms i, j and k is valid (i.e. the atoms are distinct and the bond lengths are less than 3.0 Angstroms)
        input: atomic indices i, j, k (int)
        output: True if the bond angle is valid, False otherwise (bool)"""
        return len({i, j, k}) == 3 and self.bond_length(i, j) < 3.0 and self.bond_length(j, k) < 3.0 and self.bond_angle(i, j, k) > 90.0

    def print_bond_angles(self):
        """Print the bond angles between all distinct triplets of atoms"""
        print('===Bond Angle===')
        for i in range(self.n):
            for j in range(i+1, self.n):
                for k in range(j+1, self.n):
                    for tup in [(i, j, k), (j, i, k), (i, k, j)]:
                        if self.is_valid_angle(*tup):
                            print('{:3d} - {:3d} - {:3d}: {:12.5f} degrees'.format(*tup, self.bond_angle(*tup)))
                            break
        
    def out_of_plane_angle(self, i: int, j: int, k: int, l: int) -> float:
        """Calculate the out-of-plane angle between atoms i, j, k and l
        input: atomic indices i, j, k, l (int)
        output: out-of-plane angle in degrees (float)"""
        cross = np.cross(self.bond_unit_vector(k, j), self.bond_unit_vector(k, l))
        res = np.dot(cross / np.sin(np.radians(self.bond_angle(j, k, l))), self.bond_unit_vector(k, i))
        assert(np.abs(res) < 1 + 1e-7)
        res = np.sign(res) if np.abs(res) > 1 else res
        return np.degrees(np.arcsin(res))

    def is_valid_oop_angle(self, i: int, j: int, k: int, l: int) -> bool:
        """Check if the out-of-plane angle between atoms i, j, k and l is valid (i.e. the atoms are distinct, the bond angles are valid and the bond lengths are less than 3.0 Angstroms)
        input: atomic indices i, j, k, l (int)
        output: True if the out-of-plane angle is valid, False otherwise (bool)"""
        return len({i, j, k, l}) == 4 and self.is_valid_angle(j, k, l) and self.bond_length(k, i) < 3.0 and self.bond_angle(j, k, l) < 180.0

    def print_oop_angles(self):
        """Print the out-of-plane angles between all distinct quadruplets of atoms"""
        print('===Out of Plane Angle===')
        for j in range(self.n):
            for k in range(j+1, self.n):
                for l in range(k+1, self.n):
                    for tup in [(j, k, l), (k, j, l), (j, l, k)]:
                        if self.is_valid_angle(*tup):
                            for i in range(self.n):
                                if i not in tup and self.is_valid_oop_angle(i, *tup):
                                    print('{:3d} - {:3d} - {:3d} - {:3d}: {:12.5f} degrees'.format(i, *tup, self.out_of_plane_angle(i, *tup)))

    def dihedral_angle(self, i: int, j: int, k: int, l: int) -> float:
        """Calculate the dihedral angle between atoms i, j, k and l
        input: atomic indices i, j, k, l (int)
        output: dihedral angle in degrees (float)"""
        n1 = np.cross(self.bond_unit_vector(j, i), self.bond_unit_vector(j, k))
        n2 = np.cross(self.bond_unit_vector(k, j), self.bond_unit_vector(k, l))
        val = np.dot(n1, n2)
        val /= np.sin(np.radians(self.bond_angle(i, j, k))) * np.sin(np.radians(self.bond_angle(j, k, l)))
        assert(np.abs(val) < 1 + 1e-7)
        val = np.sign(val) if np.abs(val) > 1 else val
        return np.degrees(np.arccos(val))
    
    def is_valid_dihedral_angle(self, i: int, j: int, k: int, l: int) -> bool:
        """Check if the dihedral angle between atoms i, j, k and l is valid (i.e. the atoms are distinct, the bond angles are valid and the bond lengths are less than 3.0 Angstroms)
        input: atomic indices i, j, k, l (int)
        output: True if the dihedral angle is valid, False otherwise (bool)"""
        return len({i, j, k, l}) == 4 \
            and (self.is_valid_angle(i, j, k) or self.is_valid_angle(i, k, j))\
            and (self.is_valid_angle(j, k, l) or self.is_valid_angle(k, j, l))
    
    def print_dihedral_angles(self):
        """Print the dihedral angles between all distinct quadruplets of atoms"""
        print('===Dihedral Angle===')
        for j in range(self.n):
            for k in range(j+1, self.n):
                for i in range(self.n):
                    for l in range(i+1, self.n):
                        if self.is_valid_dihedral_angle(i, j, k, l):
                            print('{:3d} - {:3d} - {:3d} - {:3d}: {:12.5f} degrees'.format(i, j, k, l, self.dihedral_angle(i, j, k, l)))

    def center_of_mass(self) -> np.ndarray:
        """Calculate the center of mass of the molecule
        output: center of mass in Bohr (np.ndarray)"""
        return np.sum([ELEMENTS[charge].mass * coord for charge, coord in zip(self.charges, self.coordinates)], axis=0) / np.sum([ELEMENTS[charge].mass for charge in self.charges])

    def moment_of_inertia(self) -> np.ndarray:
        """Calculate the moment of inertia of the molecule
        output: moment of inertia in amu bohr^2 (np.ndarray)"""
        weights = np.array([ELEMENTS[charge].mass for charge in self.charges])
        trans_coords = self.coordinates - self.center_of_mass()
        tensor = - np.einsum('i, ij, ik -> jk', weights, trans_coords, trans_coords)
        np.fill_diagonal(tensor, tensor.diagonal() + np.trace(np.abs(tensor)))
        return np.linalg.eigvalsh(tensor)

    def print_moment_of_inertia(self):
        """Print the moment of inertia of the molecule in amu bohr^2, amu angstrom^2 and g cm^2"""
        print('===Moment of Inertia===')
        print('{:3.8e} {:3.8e} {:3.8e} amu bohr^2'.format(*self.moment_of_inertia()))
        print('{:3.8e} {:3.8e} {:3.8e} amu angstrom^2'.format(*(self.moment_of_inertia() * a_0**2 / ang**2)))
        print('{:3.8e} {:3.8e} {:3.8e} g cm^2'.format(*(self.moment_of_inertia() * a_0**2 /centi**2 * amu * kilo)))

    def type_of_inertia(self) -> str:
        """Determine the type of inertia of the molecule
        output: type of inertia (str)"""
        inertia = self.moment_of_inertia()
        if np.abs(inertia[0] - inertia[1]) < 1e-4:
            return "spherical" if np.abs(inertia[1] - inertia[2]) < 1e-4 else "oblate"
        elif np.abs(inertia[0]) < 1e-4 or np.abs((inertia[1] - inertia[0]) / inertia[0]) > 1e4:
            return "linear"
        else:
            return "prolate" if np.abs(inertia[1] - inertia[2]) < 1e-4 else "asymmetric"

    def rotational_constants(self) -> np.ndarray:
        """Calculate the rotational constants of the molecule
        output: rotational constants in Hz (np.ndarray)"""
        inertia = self.moment_of_inertia() * a_0**2 * amu
        return h / (8 * np.pi**2 * inertia * c * 100)
    
    def print_solution_01(self):
        """Print the solution to the first part of the project"""
        self.print_bond_lengths()
        self.print_bond_angles()
        self.print_oop_angles()
        self.print_dihedral_angles()
        print('===Center of Mass===')
        print('{:3.5f} {:3.5f} {:3.5f}'.format(*self.center_of_mass()))
        self.print_moment_of_inertia()
        print(f'Type of inertia: {self.type_of_inertia()}')
        print(f'Rotational constants in cm^-1: {self.rotational_constants()}')
        print(f'Rotational constants in MHz: {self.rotational_constants() * c / centi / mega}')

if __name__ == '__main__':
    mol = Molecule()
    mol.read_dat('input/acetaldehyde.dat')
    mol.print_solution_01()