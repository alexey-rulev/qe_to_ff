
import numpy as np
from ase import Atoms

def random_vacancy(atoms: Atoms, species: str, rng: np.random.Generator) -> Atoms:
    idx = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == species]
    if not idx:
        raise ValueError(f"No atoms of species '{species}' to remove.")
    remove = rng.choice(idx)
    out = atoms.copy()
    del out[remove]
    return out
