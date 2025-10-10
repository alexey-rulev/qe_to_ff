
from typing import Dict, Optional, Sequence
from pathlib import Path
from ase.io import write

DEFAULT_CONTROL = dict(calculation="scf", prefix="calc", pseudo_dir="./", verbosity="low")
DEFAULT_SYSTEM = dict(occupations="smearing", smearing="mp", degauss=0.02)
DEFAULT_ELECTRONS = dict(conv_thr=1e-8)
DEFAULT_KPTS = (1, 1, 1)

def write_qe_input(
    atoms,
    path: Path,
    pseudo_map: Optional[Dict[str, str]] = None,
    ecutwfc: Optional[float] = None,
    ecutrho: Optional[float] = None,
    control: Optional[Dict] = None,
    system: Optional[Dict] = None,
    electrons: Optional[Dict] = None,
    kpts: Optional[Sequence[int]] = None,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    params = dict(
        pseudopotentials=pseudo_map or {},
        kpts=kpts or DEFAULT_KPTS,
        input_data={
            "CONTROL": {**DEFAULT_CONTROL, **(control or {})},
            "SYSTEM": {
                **DEFAULT_SYSTEM,
                **({} if ecutwfc is None else {"ecutwfc": ecutwfc}),
                **({} if ecutrho is None else {"ecutrho": ecutrho}),
                **(system or {}),
            },
            "ELECTRONS": {**DEFAULT_ELECTRONS, **(electrons or {})},
        },
    )
    symbols_needed = sorted(set(atoms.get_chemical_symbols()))
    missing = [s for s in symbols_needed if s not in (pseudo_map or {})]
    if missing:
        raise ValueError(
            f"Missing pseudopotentials for species: {missing}. "
            f"Provided keys: {sorted((pseudo_map or {}).keys())}"
        )
    write(str(path), atoms, format="espresso-in", **params)
