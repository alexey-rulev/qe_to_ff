
from typing import Optional
import warnings

try:
    import torch  # noqa: F401
    from mattersim.forcefield.potential import MatterSimCalculator  # ASE-compatible interface
    HAS_MS = True
except Exception as e:
    HAS_MS = False
    _MS_IMPORT_ERR = e

def build_calculator(ckpt_path: Optional[str] = None, device: str = "cuda", include_forces: bool = True):
    if not HAS_MS:
        warnings.warn(f"MatterSim not available: {_MS_IMPORT_ERR}")
        return None
    try:
        calc = MatterSimCalculator(
            model_path=ckpt_path or "MatterSim-v1.0.0-5M.pth",
            device=device,
            compute_forces=include_forces,
        )
        return calc
    except Exception as e:
        warnings.warn(f"Failed to init MatterSim calculator: {e}")
        return None

def compute_energy_forces(atoms):
    try:
        e = atoms.get_potential_energy()
        f = atoms.get_forces()
        return e, f
    except Exception as e:
        warnings.warn(f"Energy/forces not available: {e}")
        return None, None
