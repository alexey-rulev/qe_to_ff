
# ff_trainer

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
ff_trainer \  --input structure.cif \  --output out_dir \  --supercell 2 2 2 \  --random-n 100 --random-sigma 0.1 \  --lattice-m 50 --lattice-span 0.05 \  --vacancy-k 20 --vacancy-species O --vacancy-sigma 0.05 \  --mattersim-ckpt MatterSim-v1.0.0-5M.pth \  --qe-pseudo Si:Si.pbe-n-kjpaw_psl.1.0.0.UPF O:O.pbe-n-kjpaw_psl.1.0.0.UPF \  --qe-ecutwfc 60 --qe-ecutrho 480
```

## Post-processing QE outputs

Use the companion CLI `ff-trainer-post` (entry point for `ff_trainer/cli_post.py`) to merge finished QE `pw.x` jobs, optionally recompute energies/forces with MatterSim, and build diagnostic plots.

Example end-to-end call:

```bash
ff-trainer-post \
  --dirs qe_submit/in qe_submit/slurm \
  --out-extxyz results/qe_runs.extxyz \
  --plot results/qe_vs_mattersim.png \
  --mattersim-ckpt MatterSim-v1.0.0-5M.pth \
  --device cuda \
  --skip-missing-forces
```

What this does:
- Crawls every provided `--dirs` path for `*.out` files (individual files are also accepted).
- Parses each output, skipping ones without QE forces if `--skip-missing-forces` is set.
- Optionally attaches a MatterSim calculator to compute energies/forces (`--no-forces` disables the force re-evaluation).
- Writes the merged dataset to `--out-extxyz` and, if requested, saves a QE vs MatterSim scatter plot.
- Prints how many frames were written plus a count of skipped outputs (helpful when some calculations did not converge to forces).

## Configuration tags in outputs

Generated `*.extxyz` files include per-frame metadata that you can filter on when training models:
- `group="random"` – random-displacement supercells from the primary sampling loop.
- `group="lattice"` plus `scale=<float>` – isotropically strained cells from the lattice scan series.
- `group="vacancy"` – configurations where one atom of `vacancy_species` was removed and locally perturbed.

Use these tags to create balanced training splits or to focus diagnostics on a specific regime.

## CLI tags and options

### `ff-trainer` (dataset generation)

| Tag | Description |
| --- | --- |
| `--input` | Input structure (CIF/POSCAR/XYZ) used as the seed geometry. |
| `--output` | Destination directory; subfolders for random/lattice/vacancies, QE inputs, plots, etc. |
| `--supercell a b c` | Repeat factors for the base structure (default `2 2 2`). |
| `--random-n` | Number of random-displacement configurations. |
| `--random-sigma` | Std. dev. of random displacements in Å. |
| `--lattice-m` | Number of lattice points in the ±`lattice-span` scan. |
| `--lattice-span` | Fractional strain applied symmetrically (default 0.05 = ±5%). |
| `--vacancy-k` | Number of vacancy structures to generate. |
| `--vacancy-species` | Chemical symbol for the atom removed in vacancy configs. |
| `--vacancy-sigma` | Random displacement sigma applied after carving the vacancy. |
| `--mattersim-ckpt` | MatterSim checkpoint path; set to `None` to skip ML energy evaluation. |
| `--device` | `cuda` or `cpu` for MatterSim inference. |
| `--cutoff-dist` | Cutoff (Å) for pair-distance analysis plots. |
| `--qe-pseudo Symbol:File ...` | Map of pseudopotentials passed to QE input writer (e.g., `Si:Si.UPF`). |
| `--qe-ecutwfc` | Plane-wave cutoff (Ry) in QE inputs. |
| `--qe-ecutrho` | Charge-density cutoff (Ry). |
| `--rng-seed` | Seed for NumPy random generator to keep runs reproducible. |

### `ff-trainer-post` (QE consolidation)

| Tag | Description |
| --- | --- |
| `--dirs PATH [...]` | Directories and/or specific `.out` files to scan for QE outputs. |
| `--out-extxyz FILE` | Destination extended XYZ with consolidated frames. |
| `--plot FILE` | Optional PNG path for QE vs MatterSim scatter plot. |
| `--mattersim-ckpt FILE` | MatterSim checkpoint for recomputing energies/forces; omit to skip ML comparisons. |
| `--device {cuda,cpu}` | Execution device for MatterSim when a checkpoint is provided. |
| `--no-forces` | Do not store MatterSim forces (energy-only comparison). |
| `--skip-missing-forces` | Ignore QE runs where forces failed to parse, preventing empty or invalid frames. |
