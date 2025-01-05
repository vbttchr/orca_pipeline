import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.cm as cm
from orca_pipeline.chemistry import Reaction
from orca_pipeline.constants import HARTREE_TO_KCAL_MOL, kb

# from ..constants import HARTREE_TO_KCAL_MOL


# ...existing code...

def plot_reaction_profile(Reactions: list, show: bool = True, save: bool = False, type="free_energy"):
    # TODO Change ploting to use energy diagram package
    """
    Plot the reaction profile of a reaction.
    Parameters
    ----------
    Reactions : list
        List of ordered Reaction objects.
        type : str
        Type of energy to plot. Options are "free_energy","enthalpy" or "inner_energy).

    """

    # total_steps = len(Reactions)*3 - len(Reactions) - 1

    steps = []
    energies_plot = []
    y_label = ""

    for count, reaction in enumerate(Reactions):

        energies = reaction.energies
        for index, row in energies.iterrows():
            if count != 0 and index == 0:
                continue
            steps.append(row['step'])
            if type == "free_energy":

                energies_plot.append(
                    row['single_point_energy'] + row['free_energy_correction'] + row['solvation_correction'])
                y_label = "Free Energy (kcal/mol)"
            elif type == "enthalpy":
                inner_energy = row['single_point_energy'] + \
                    row["inner_energy_correction"]
                enthalpy = inner_energy + row["temperature"] * kb
                label = "Enthalpy (kcal/mol)"

                energies_plot.append(enthalpy)
            elif type == "inner_energy":
                inner_energy = row['single_point_energy'] + \
                    row["inner_energy_correction"]
                energies_plot.append(inner_energy)
                y_label = "Inner Energy (kcal/mol)"
            else:
                raise ValueError(
                    "Invalid type. Options are 'free_energy','enthalpy' or 'inner_energy'.")

    sns.lineplot(x=steps, y=energies_plot)
    plt.xlabel('Reaction Coordinate')
    plt.ylabel(y_label)
    if show:
        plt.show()
    if save:
        plt.savefig(f'{Reactions[0].name}_reaction_profile.png')


def parse_xyz_energies(xyz_file):
    energies = []
    with open(xyz_file, 'r') as f:
        for line in f:
            if 'Coordinates from ORCA-job' in line:
                match = re.search(r'E\s+([-+]?\d*\.\d+|\d+)', line)
                if match:

                    energies.append(float(match.group(1)))
    return energies


# ...existing code...
def plot_optimization(energies, images: int = 12, show: bool = True, save: bool = False):
    """
    Plot the optimization of a NEB calculation.
    Parameters
    ----------
    energies : list
        List of energies from the NEB calculation.
    images : int    
        Number of images in the NEB calculation.
    """

    steps = images + 2  # Number of images + 2 endpoints
    total_steps = len(energies) // steps
    # Exclude extremes of the palette
    colors = cm.Blues(np.linspace(0.2, 0.8, total_steps))
    x = np.arange(0, steps)
    for i in range(0, len(energies), steps):
        relative_index = i // steps
        if relative_index < total_steps:
            relative_energies = [(energy - energies[i])*HARTREE_TO_KCAL_MOL
                                 for energy in energies[i:i + steps]]
            sns.lineplot(x=x, y=relative_energies,
                         color=colors[relative_index])

    if show:
        plt.show()
    if save:
        plt.savefig('neb_plot.png')
# ...existing code...


if __name__ == "__main__":
    xyz_file = '/home/joel/orca_pipeline/orca_pipeline/plotting/neb-TS_MEP_ALL_trj.xyz'
    energies = parse_xyz_energies(xyz_file)
    plot_optimization(energies, images=16, show=True, save=False)
