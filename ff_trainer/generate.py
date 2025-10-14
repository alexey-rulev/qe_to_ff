
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import numpy as np
from ase.io import read
from ase.io.espresso import read_espresso_in
from ase import Atoms
from ase.build import make_supercell
from ase.geometry import cell_to_cellpar
from .geometry import random_displacements, lattice_scan_series
from .vacancies import random_vacancy
from .mattersim_wrapper import build_calculator, compute_energy_forces
from .io_qe import write_qe_input
from .io_xyz import write_extxyz
from .analysis import violin_pair_distance_plot, energy_distribution_plot, energy_vs_lattice_plot

def generate_all(
    input_structure: str,
    output_dir: str,
    supercell: Tuple[int,int,int] = (2,2,2),
    random_n: int = 100,
    random_sigma: float = 0.1,
    lattice_m: int = 50,
    lattice_span: float = 0.05,
    vacancy_k: int = 20,
    vacancy_species: str = "O",
    vacancy_sigma: float = 0.05,
    mattersim_ckpt: Optional[str] = "MatterSim-v1.0.0-5M.pth",
    device: str = "cuda",
    cutoff_dist: float = 4.0,
    qe_pseudo: Optional[Dict[str, str]] = None,
    qe_ecutwfc: Optional[float] = 60.0,
    qe_ecutrho: Optional[float] = 480.0,
    rng_seed: int = 42,
):
    out = Path(output_dir)
    (out / "random" / "qe").mkdir(parents=True, exist_ok=True)
    (out / "lattice_scan" / "qe").mkdir(parents=True, exist_ok=True)
    (out / "vacancies" / "qe").mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(rng_seed)
    with open(input_structure, 'r') as in_file:
        base = read_espresso_in(in_file)
    base_sc = make_supercell(base, np.diag(supercell))
    a0 = cell_to_cellpar(base_sc.cell)[0]

    calc = build_calculator(mattersim_ckpt, device=device, include_forces=True)
    def eval_energy_forces(at):
        at = at.copy()
        if calc is not None:
            at.calc = calc
        e, f = compute_energy_forces(at)
        return e, f

    # Random set
    random_atoms, random_E, random_F = [], [], []
    for i in range(random_n):
        at = random_displacements(base_sc, random_sigma, rng)
        e, f = eval_energy_forces(at)
        random_atoms.append(at)
        random_E.append(e if e is not None else np.nan)
        random_F.append(f if f is not None else None)
        write_qe_input(at, out / "random" / "qe" / f"random_{i:04d}.in", pseudo_map=qe_pseudo, ecutwfc=qe_ecutwfc, ecutrho=qe_ecutrho)
    write_extxyz(random_atoms, random_E, random_F, out / "random" / "random.extxyz")

    # Lattice scan
    lattice_series = lattice_scan_series(base_sc, lattice_span, lattice_m)
    lattice_atoms, lattice_E, lattice_F, lattice_scales = [], [], [], []
    for i, (s, at) in enumerate(lattice_series):
        e, f = eval_energy_forces(at)
        lattice_atoms.append(at)
        lattice_E.append(e if e is not None else np.nan)
        lattice_F.append(f if f is not None else None)
        lattice_scales.append(s)
        write_qe_input(at, out / "lattice_scan" / "qe" / f"lattice_{i:04d}.in", pseudo_map=qe_pseudo, ecutwfc=qe_ecutwfc, ecutrho=qe_ecutrho)
    write_extxyz(lattice_atoms, lattice_E, lattice_F, out / "lattice_scan" / "lattice_scan.extxyz")

    # Vacancies
    vac_atoms, vac_E, vac_F = [], [], []
    for i in range(vacancy_k):
        at = random_vacancy(base_sc, vacancy_species, rng)
        at = random_displacements(at, vacancy_sigma, rng)
        e, f = eval_energy_forces(at)
        vac_atoms.append(at)
        vac_E.append(e if e is not None else np.nan)
        vac_F.append(f if f is not None else None)
        write_qe_input(at, out / "vacancies" / "qe" / f"vacancy_{i:04d}.in", pseudo_map=qe_pseudo, ecutwfc=qe_ecutwfc, ecutrho=qe_ecutrho)
    write_extxyz(vac_atoms, vac_E, vac_F, out / "vacancies" / "vacancies.extxyz")

    # Combined
    all_atoms = random_atoms + lattice_atoms + vac_atoms
    all_E = random_E + lattice_E + vac_E
    all_F = random_F + lattice_F + vac_F
    info = (
        [{"group": "random"} for _ in random_atoms]
        + [{"group": "lattice", "scale": s} for s in lattice_scales]
        + [{"group": "vacancy"} for _ in vac_atoms]
    )
    write_extxyz(all_atoms, all_E, all_F, out / "all.extxyz", info_list=info)

    # Plots (representative configs for violins to keep figures readable)
    reps, labels = [], []
    if random_atoms: reps.append(random_atoms[0]); labels.append("random_0")
    if lattice_atoms: reps.append(lattice_atoms[0]); labels.append("lattice_0")
    if vac_atoms: reps.append(vac_atoms[0]); labels.append("vacancy_0")
    if reps:
        from .analysis import violin_pair_distance_plot
        violin_pair_distance_plot(reps, labels, out / "plots" / "pair_violin.png", cutoff=cutoff_dist)

    # Energy hists
    energy_by_group = {}
    if random_E: energy_by_group["random"] = [float(x) for x in random_E if x == x]
    if lattice_E: energy_by_group["lattice"] = [float(x) for x in lattice_E if x == x]
    if vac_E: energy_by_group["vacancy"] = [float(x) for x in vac_E if x == x]
    if energy_by_group:
        energy_distribution_plot(energy_by_group, out / "plots" / "energy_hist.png")

    # Energy vs a
    finite = [(s, e) for s, e in zip(lattice_scales, lattice_E) if e == e]
    if finite:
        scales, energies = zip(*finite)
        energy_vs_lattice_plot(list(scales), list(energies), a0, out / "plots" / "energy_vs_a.png")

    return str(out)
