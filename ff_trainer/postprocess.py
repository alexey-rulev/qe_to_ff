
from __future__ import annotations
from typing import Iterable, List, Optional, Dict, Any
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from ase.io import write
from ase import Atoms

# Use the user's reader directly
from .qe_out_reader import read_qe_out_to_atoms

from .mattersim_wrapper import build_calculator, compute_energy_forces

def _qe_energy(at: Atoms) -> Optional[float]:
    try:
        # prefer ASE calculator energy if attached
        return float(at.get_potential_energy())
    except Exception:
        # fallback from info if calculator missing
        for key in ("energy", "total_energy", "free_energy"):
            if key in at.info:
                try:
                    return float(at.info[key])
                except Exception:
                    pass
    return None

def collect_and_compare(
    search_dirs: Iterable[str],
    out_extxyz: str,
    mattersim_ckpt: Optional[str] = None,
    device: str = "cuda",
    include_forces: bool = True,
    plot_path: Optional[str] = None,
    skip_missing_forces: bool = False,
) -> Dict[str, Any]:
    """
    Scan directories for QE *.out, parse with qe_out_reader.read_qe_out_to_atoms,
    aggregate into a single extended XYZ, compute MatterSim energies (optional),
    and plot MatterSim vs QE energies.

    If skip_missing_forces is True, configurations lacking QE forces are ignored.
    """
    # Discover .out files
    files: List[Path] = []
    for d in search_dirs:
        p = Path(d)
        if p.is_file() and p.suffix == ".out":
            files.append(p.resolve())
        elif p.is_dir():
            files.extend(sorted(q.resolve() for q in p.rglob("*.out")))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError("No .out files found under: " + ", ".join(map(str, search_dirs)))

    # Optional MatterSim calc
    calc = None
    if mattersim_ckpt:
        calc = build_calculator(mattersim_ckpt, device=device, include_forces=include_forces)

    frames: List[Atoms] = []
    qe_E: List[float] = []
    ms_E: List[float] = []

    skipped: List[str] = []

    for f in files:
        at = read_qe_out_to_atoms(str(f))
        e_qe = _qe_energy(at)

        has_forces = False
        calc_results = getattr(getattr(at, "calc", None), "results", None)
        if isinstance(calc_results, dict) and calc_results.get("forces") is not None:
            has_forces = True

        if skip_missing_forces and not has_forces:
            skipped.append(str(f))
            continue

        e_ms = None
        f_ms = None
        if calc is not None:
            at_ms = at.copy()
            at_ms.calc = calc
            e_ms, f_ms = compute_energy_forces(at_ms)

        frames.append(at)
        qe_E.append(e_qe if e_qe is not None else np.nan)
        ms_E.append(e_ms if e_ms is not None else np.nan)

    # Write combined XYZ
    out_path = Path(out_extxyz).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write(str(out_path), frames, format="extxyz")

    # Plot scatter
    plot_file = None
    if plot_path is not None:
        plot_file = Path(plot_path).resolve()
        plot_file.parent.mkdir(parents=True, exist_ok=True)
        x = np.array([e for e in qe_E if e == e], float)
        y = np.array([e for e in ms_E if e == e], float)
        n = min(len(x), len(y))
        x, y = x[:n], y[:n]
        if n > 0:
            plt.figure()
            plt.scatter(x, y, s=12)
            lo = float(min(x.min(), y.min()))
            hi = float(max(x.max(), y.max()))
            try:
                coeffs = np.polyfit(x, y, 1)
                fit = np.poly1d(coeffs)
                yhat = fit(x)
                ss_res = np.sum((y - yhat)**2)
                ss_tot = np.sum((y - np.mean(y))**2)
                r2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
                ttl = f"MatterSim vs QE Energy (R^2 = {r2:.3f})"
            except Exception:
                ttl = "MatterSim vs QE Energy"
            plt.title(ttl)
            plt.xlabel("QE energy (eV)")
            plt.ylabel("MatterSim energy (eV)")
            plt.tight_layout()
            plt.savefig(plot_file, dpi=200)
            plt.close()

    return {
        "n": len(frames),
        "files": [str(f) for f in files],
        "extxyz": str(out_path),
        "plot": str(plot_file) if plot_path is not None else None,
        "skipped": skipped,
    }
