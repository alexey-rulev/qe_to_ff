
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
 
## Workflow
The script is intended for building the initial rough training set for fine tuning. The intended workflow is following:
- You get your (optimized) unit cell configuration
- The script generates set of configurations:
  - supercells with random displacements
  - undisturbed cell with varied lattice parameter (EOS)
  - supercell with specified vacancies and random displacements
- The generated configuraitons carry mostly structural information. For each configuration, user must prepare a template with all the parameters like cutoff energies, Hubbard parameters etc.
- The template, the configurations and submit script (must be adapted to your cluster) are uploaded to the computational cluster
- The submit script takes most parameters from the template, and only nat, ntyp and cell and atom geometry from configuration files (so for example kpoints should be supplied in template), and submits single-point calculations.
- the results are collected and post-processed with a post-process script, that takes all the relaxed geometries, calculated forces and energies and writes it to a single extended xyz file, that can later be used for fine tuning the model.

## Quickstart
```bash
pip install -e .
```
## Generate configurations:
```bash
ff_trainer \  --input structure.cif \  --output out_dir \  --supercell 2 2 2 \  --random-n 100 --random-sigma 0.1 \  --lattice-m 50 --lattice-span 0.05 \  --vacancy-k 20 --vacancy-species O --vacancy-sigma 0.05 \  --mattersim-ckpt MatterSim-v1.0.0-5M.pth \  --qe-pseudo Si:Si.pbe-n-kjpaw_psl.1.0.0.UPF O:O.pbe-n-kjpaw_psl.1.0.0.UPF \  --qe-ecutwfc 60 --qe-ecutrho 480
```
<<<<<<< HEAD

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
=======
## Submitting calculations
there is a script submit_qe_configs.py that submits all the configurations as jobs to SLURM scheduler. You need to add the script and a QE pw.in file template with all the parameters in the directory. The script copies the geometry, nat and ntyp from the generated configurations into the template and submits jobs. Modify if needed to match your infrastructure. Submit by running:
```bash
python submit_qe_configs.py --template template.in --inputs "*.in"
```
assuming that the submit script, template and generated .in files are in the same directory. Be careful with the paths to pseudopotentials.
## Post-process
Post-processing script reads all .out files, and collects them into a single extxyz file (and runs some preliminary statistics):
```bash
ff-trainer-post --dirs <dir with .out files> --plot ./ms_vs_qe.png --mattersim-ckpt mattersim-v1.0.0-5M.pth --device cpu --out-extxyz vac.extxyz
```
the .out files may be in the subdirectories, the script will find them. Caution with slurm .out files (or any other .out files), they will confuse the script, so they should be deleted.


>>>>>>> fb077f4ddb7b0ed2b0ec08d794fd32f59db520fc
