
# mattergen

Generate configuration sets for DFT fine‑tuning of an MD force field (e.g., MatterSim), starting from a given structure.
It reproduces the workflow in your Jupyter notebook but organizes it as a modular, reusable package.

## Features
- Random‑displacement supercells (default: 2×2×2, 100 structures)
- Lattice scan at fixed atomic positions (±5% by default, 50 steps configurable)
- Vacancy configs by removing a random atom of a given species (default: O, 2×2×2, 20 structures)
- Energy/forces via pretrained MatterSim (default: `MatterSim-v1.0.0-5M.pth`) if available
- Writes **Quantum ESPRESSO** input files for each config
- Aggregates all configurations with energies/forces into a single **extended XYZ**
- Plots:
  - Violin plots of pair distance distributions (PBC-aware, cutoff configurable)
  - Distributions of energies per configuration group
  - Total potential energy vs. lattice parameter for the lattice-scan set

## Quickstart
```bash
pip install -e .
mattergen \  --input structure.cif \  --output out_dir \  --supercell 2 2 2 \  --random-n 100 --random-sigma 0.1 \  --lattice-m 50 --lattice-span 0.05 \  --vacancy-k 20 --vacancy-species O --vacancy-sigma 0.05 \  --mattersim-ckpt MatterSim-v1.0.0-5M.pth \  --qe-pseudo Si:Si.pbe-n-kjpaw_psl.1.0.0.UPF O:O.pbe-n-kjpaw_psl.1.0.0.UPF \  --qe-ecutwfc 60 --qe-ecutrho 480
```
