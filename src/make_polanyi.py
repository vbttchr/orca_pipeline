import sys

from polanyi.io import read_xyz, write_xyz, get_xyz_string
from polanyi.interpolation import interpolate_geodesic
from polanyi.workflow import opt_ts, opt_ts_python, opt_ts_ci_python
from rdkit import Chem
from os import PathLike
import numpy as np
from typing import Optional, Union
import os
import glob

current=os.getcwd()

def gen_ts(
    elem_ts: list[str],
    coord_r_complex: np.ndarray,
    coord_p_complex: np.ndarray,
    charge: int,
    solvent: str=None,
    coupling: float = 0.001,
    guess_coord_ts: Optional[np.ndarray] = None,
    guess_image :Optional[int] = None,
    use_xtb_python: bool = False,
    use_ci: bool = False,
    images: int =9,
    paths_topo: Optional[list[Union[str, PathLike]]] = None,
    paths_e_shift: Optional[list[Union[str, PathLike]]] = None,
    path_ts: Optional[Union[str, PathLike]] = None,
    save_guess: bool = True
    ) -> np.ndarray:
    """Generate TS with polanyi
    Args:
        elem_ts: elements of TS
        coord_r_complex: coordinates of reactant complex or intermediate
        coord_p_complex: coordinates of product complex or intermediate
        charge: total charge of the system
        solvent: name of the solvent
        coupling: coupling term (default: 0.001)
        use_xtb_python: whether to use the Polanyi functions with xtb-python (False by default)
        use_ci: whether to use the conical intersection optimisation (False by default)
        paths_topo: list of folders for topology calculations of xTB command line on ground states
        paths_e_shift: list of folders for energy shift calculations of xTB command line on ground states
        path_ts: folder for TS calculation of xTB command line
    Returns:
        coordinates of the TS
    """
    # Define images number for interoplation
    n_images = images 
    idx_taken_image = n_images // 2
    if guess_image is not None:
        if guess_image < n_images:
            idx_taken_image = guess_image
            
     
    if guess_coord_ts is None:
    # Interpolate guess structure between reactant and product complex
        path = interpolate_geodesic(
            elem_ts, [coord_r_complex, coord_p_complex], n_images=n_images
        )
        coord_ts_guess = path[idx_taken_image]

        
            #write_xyz(f"{images}+_{coupling}_ts_guess.xyz", elem_ts, coord_ts_guess)
    else:
        coord_ts_guess = guess_coord_ts
    # Generate keywords for GFN2-XTB calculation
    if use_xtb_python or use_ci:
        if solvent != None:
            keywords_xtb = {"charge": charge}
        else:
            keywords_xtb = {"charge": charge, 'solvent': solvent}
    else:
        if solvent != None:
            keywords_xtb = [f"--chrg {charge}", f"--alpb {solvent}" ]
        else:
            keywords_xtb = [f"--chrg {charge}"]

    # Optimise
    if use_ci:
        coord_opt = opt_ts_ci_python(
            elem_ts,
            [coord_r_complex, coord_p_complex],
            coord_ts_guess,
            kw_calculators=keywords_xtb,
        )
        return coord_opt
    
    elif use_xtb_python:
        results = opt_ts_python(
            elem_ts,
            [coord_r_complex, coord_p_complex],
            coord_ts_guess,
            kw_calculators=keywords_xtb,
            kw_opt={"coupling": coupling},
        )

    else:
        results = opt_ts(
            elem_ts,
            [coord_r_complex, coord_p_complex],
            coord_ts_guess,
            kw_calculators={"keywords": keywords_xtb, "paths": paths_topo},
            kw_opt={"keywords": keywords_xtb, "path": path_ts, "coupling": coupling},
            kw_shift={
                "keywords_ff": keywords_xtb,
                "keywords_sp": keywords_xtb,
                "paths": paths_e_shift,
            },
        )
        if save_guess:
            return results.coordinates_opt, coord_ts_guess
        

    return results.coordinates_opt

def get_charge(all_smiles: list[str]) -> int:
    """Calculate the total molecular charge
    Args:
        all_smiles: SMILES of the reactants
    Returns:
        total molecular charge of the reactants
    """
    tot_chg = 0
    for smiles in all_smiles:
        mol = Chem.MolFromSmiles(smiles)
        chg = Chem.rdmolops.GetFormalCharge(mol)
        tot_chg += chg
    return tot_chg



if __name__ == "__main__":
    # Define the reactant complex and product complex
    # The coordinates of the reactant complex and product complex

    name = "test"
    ele,coord_int= read_xyz("tst/educt.xyz")
    ele,coord_prod= read_xyz("tst/product.xyz")
    ts_coord, guess_coord = gen_ts(elem_ts=ele, coord_r_complex=coord_int, coord_p_complex=coord_prod, charge=-1, solvent='thf',coupling=0.001, images=9,save_guess=True)
    write_xyz(name+'_ts_1_9_images_0.0001.xyz', ele, ts_coord)
    write_xyz(name+'_ts_19_images_0.0001_guess.xyz', ele, guess_coord)


                 
                     
                    
                         
                         
            

            

            


