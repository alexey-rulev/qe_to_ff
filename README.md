
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


