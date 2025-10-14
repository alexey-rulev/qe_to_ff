
import numpy as np
from ase import Atoms

def random_displacements(atoms: Atoms, sigma: float, rng: np.random.Generator) -> Atoms:
    out = atoms.copy()
    out.positions += rng.normal(0.0, sigma, size=out.positions.shape)
    return out

def scale_lattice_keep_frac_positions(atoms: Atoms, scale: float) -> Atoms:
    out = atoms.copy()
    frac = out.get_scaled_positions()
    cell = out.cell * scale
    out.set_cell(cell, scale_atoms=False)
    out.set_scaled_positions(frac)
    return out

def lattice_scan_series(atoms: Atoms, span: float, m: int):
    import numpy as np
    scales = np.linspace(1.0 - span, 1.0 + span, m)
    return [(s, scale_lattice_keep_frac_positions(atoms, s)) for s in scales]
