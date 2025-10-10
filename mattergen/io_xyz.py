
from typing import List, Optional, Dict, Any
from pathlib import Path
import numpy as np
from ase.io import write

def write_extxyz(
    atoms_list: List,
    energies: List[Optional[float]],
    forces: List[Optional[np.ndarray]],
    path: Path,
    info_list: Optional[List[Dict[str, Any]]] = None,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if info_list is None:
        info_list = [{} for _ in atoms_list]

    frames = []
    for at, e, f, inf in zip(atoms_list, energies, forces, info_list):
        at = at.copy()
        if e is not None:
            at.info["energy"] = float(e)
        if f is not None:
            at.arrays["forces"] = f
        at.info.update(inf or {})
        frames.append(at)
    write(str(path), frames, format="extxyz")
