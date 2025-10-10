
from typing import List, Dict, Tuple
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from ase import Atoms
from ase.neighborlist import NeighborList

def pair_distances(atoms: Atoms, cutoff: float = 4.0):
    n = len(atoms)
    cutoffs = [cutoff/2]*n
    nl = NeighborList(cutoffs, skin=0.0, self_interaction=False, bothways=True)
    nl.update(atoms)

    species = atoms.get_chemical_symbols()
    buckets = {}
    for i in range(n):
        idxs, offs = nl.get_neighbors(i)
        for j, off in zip(idxs, offs):
            if j <= i:
                continue
            vec = atoms.positions[j] + np.dot(off, atoms.get_cell()) - atoms.positions[i]
            d = np.linalg.norm(vec)
            if d <= cutoff:
                pair = tuple(sorted((species[i], species[j])))
                buckets.setdefault(pair, []).append(d)

    return {k: np.array(v) for k, v in buckets.items()}

def violin_pair_distance_plot(atoms_list: List[Atoms], labels: List[str], out_png: Path, cutoff: float = 4.0):
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    for atoms, label in zip(atoms_list, labels):
        dists = pair_distances(atoms, cutoff=cutoff)
        pairs = sorted(dists.keys())
        data = [dists[p] for p in pairs if len(dists[p]) > 0]
        names = [f"{a}-{b}" for (a,b) in pairs if len(dists[(a,b)]) > 0]

        plt.figure()
        plt.violinplot(data, showmeans=True, showextrema=True, showmedians=True)
        plt.xticks(range(1, len(names)+1), names, rotation=45, ha="right")
        plt.ylabel("Distance (Å)")
        plt.title(f"Pair distance distribution: {label}")
        plt.tight_layout()
        plt.savefig(out_png.parent / f"pair_violin_{label}.png", dpi=200)
        plt.close()

def energy_distribution_plot(energies_by_group: Dict[str, List[float]], out_png: Path):
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    for label, vals in energies_by_group.items():
        plt.figure()
        plt.hist(vals, bins=30)
        plt.xlabel("Energy (eV)")
        plt.ylabel("Count")
        plt.title(f"Energy distribution: {label}")
        plt.tight_layout()
        plt.savefig(out_png.parent / f"energy_hist_{label}.png", dpi=200)
        plt.close()

def energy_vs_lattice_plot(scales: List[float], energies: List[float], a0: float, out_png: Path):
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    a_vals = [s * a0 for s in scales]
    plt.figure()
    plt.plot(a_vals, energies, marker="o")
    plt.xlabel("Lattice parameter a (Å)")
    plt.ylabel("Total energy (eV)")
    plt.title("Energy vs. lattice parameter (lattice scan)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
