from typing import List, Union, Tuple
import numpy as np
import os

def read_xyz(filepath: str) -> Tuple[List[str], np.ndarray]:
    """
    Reads an XYZ file and returns a list of atoms and a 3xN NumPy array of coordinates.
    """
    atoms = []
    coords = []
    with open(filepath, 'r') as file:
        lines = file.readlines()
        for line in lines[2:]:  # Skip the first two lines (header)
            parts = line.split()
            atoms.append(parts[0])
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    coords = np.array(coords).T  # Transpose to get a 3xN array
    return atoms, coords

class Molecule:
    """
    Represents a molecule.


    """

    def __init__(self, name:str, atoms: List[str], coords: np.ndarray,mult:int,charge:int,solvent:str=None,method:str="r2scan-3c") -> None:
        self.name = name
        self.atoms = atoms
        self.coords = coords
        self.mult = mult
        self.charge = charge
        self.solvent = solvent
        self.method = method


       

        
        if len(atoms) != coords.shape[1]:
            raise ValueError("Number of atoms and coordinates must match.")
        if mult < 1:
            raise ValueError("Multiplicity must be at least 1.")
    
    @classmethod

    def from_xyz(cls, filepath: str, charge:int,mult:int, solvent:str=None,name:str="mol",method:str="r2scan-3c") -> 'Molecule':
        """
        Creates a Molecule instance from an XYZ file.
        """
        if name==None:
            name = os.path.basename(filepath).split('.')[0]
        atoms, coords = read_xyz(filepath)
        return cls(name, atoms, coords, charge=charge, mult=mult, solvent=solvent,method=method)
    def __str__(self) -> str:
        return f"Molecule(name={self.name},atoms={self.atoms}, coordinates={self.coords}, Multiplicity={self.mult}, Charge={self.charge})"

    def __repr__(self) -> str:
        return f"Molecule(name={self.name},atoms={self.atoms}, coordinates={self.coords}, Multiplicity={self.mult}, Charge={self.charge})"

    
    def translate(self, vector: List[float]) -> None:
        """
        Translates the molecule by the given vector.
        """
        self.coords += np.array(vector).reshape(3, 1)

    def rotate(self, rotation_matrix: np.ndarray) -> None:
        """
        Rotates the molecule using the given 3x3 rotation matrix.
        """
        self.coords = np.dot(rotation_matrix, self.coords)
    def to_xyz(self, filepath:str =None) -> None:
        """
        Writes the molecule to an XYZ file.
        """
        if filepath is None:
            current_dir = os.getcwd()
            filepath = os.path.join(current_dir, f"{self.name}.xyz")
            
        with open(filepath, 'w') as file:
            file.write(f"{len(self.atoms)}\n")
            file.write("\n")
            for atom, coord in zip(self.atoms, self.coords.T):
                file.write(f"{atom} {coord[0]:.9f} {coord[1]:.9f} {coord[2]:.9f}\n")


class Reaction:
    """
    Represents a elementary step in a reaction.
    """

    def __init__(self, educt:Molecule,product:Molecule,transitions_state:Molecule=None,nimages:int=16,method:str="r2scan-3c") -> None:
        self.educt = educt
        self.product = product
        self.transition_state = transitions_state
        self.nimages = nimages
        self.method = method


        if educt.charge != product.charge:
            raise ValueError("Charge of educt and product must match.")
        
        if educt.mult != product.mult:
            raise ValueError("Exicited state reactions are not supported yet.")
        

        


    @classmethod

    def __str__(self) -> str:
        reactant_strs = " + ".join(f"{self.educt}" )
        product_strs = " + ".join( self.products)
        return f"{reactant_strs} => {product_strs}"

    def __repr__(self) -> str:
        reactant_strs = " + ".join(f"{self.educt}" )
        product_strs = " + ".join( self.products)
        return f"{reactant_strs} => {product_strs}"


    def from_xyz(cls, educt_filepath: str, product_filepath: str, transition_state_filepath: str = None,nimages:int=16) -> 'ElementaryStep':
        """
        Creates an ElementaryStep instance from XYZ files.
        """
        educt = Molecule.from_xyz(educt_filepath)
        product = Molecule.from_xyz(product_filepath)
        transition_state = Molecule.from_xyz(transition_state_filepath) if transition_state_filepath else None
        return cls(educt, product, transition_state)