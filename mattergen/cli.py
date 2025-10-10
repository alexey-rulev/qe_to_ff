
import argparse
from .generate import generate_all

def main():
    p = argparse.ArgumentParser(description="Generate DFT fine-tuning configuration sets and plots.")
    p.add_argument("--input", required=True, help="Input structure file (e.g., CIF/POSCAR/XYZ).")
    p.add_argument("--output", required=True, help="Output directory.")
    p.add_argument("--supercell", nargs=3, type=int, default=[2,2,2], help="Supercell repeats, e.g. 2 2 2")
    p.add_argument("--random-n", type=int, default=100)
    p.add_argument("--random-sigma", type=float, default=0.1, help="Å")
    p.add_argument("--lattice-m", type=int, default=50)
    p.add_argument("--lattice-span", type=float, default=0.05, help="±fraction")
    p.add_argument("--vacancy-k", type=int, default=20)
    p.add_argument("--vacancy-species", type=str, default="O")
    p.add_argument("--vacancy-sigma", type=float, default=0.05)
    p.add_argument("--mattersim-ckpt", type=str, default="MatterSim-v1.0.0-5M.pth")
    p.add_argument("--device", type=str, default="cuda", help="'cuda' or 'cpu'")
    p.add_argument("--cutoff-dist", type=float, default=4.0, help="Å cutoff for pair distances")
    p.add_argument("--qe-pseudo", nargs="*", default=None, help="Pairs like Si:Si.UPF O:O.UPF")
    p.add_argument("--qe-ecutwfc", type=float, default=60.0)
    p.add_argument("--qe-ecutrho", type=float, default=480.0)
    p.add_argument("--rng-seed", type=int, default=42)
    args = p.parse_args()

    pseudo_map = None
    if args.qe_pseudo:
        pseudo_map = {}
        for tok in args.qe_pseudo:
            if ":" in tok:
                s, f = tok.split(":", 1)
            else:
                raise SystemExit(f"Invalid --qe-pseudo token: {tok} (use Symbol:Filename.UPF)")
            pseudo_map[s] = f

    output = generate_all(
        input_structure=args.input,
        output_dir=args.output,
        supercell=tuple(args.supercell),
        random_n=args.random_n,
        random_sigma=args.random_sigma,
        lattice_m=args.lattice_m,
        lattice_span=args.lattice_span,
        vacancy_k=args.vacancy_k,
        vacancy_species=args.vacancy_species,
        vacancy_sigma=args.vacancy_sigma,
        mattersim_ckpt=args.mattersim_ckpt,
        device=args.device,
        cutoff_dist=args.cutoff_dist,
        qe_pseudo=pseudo_map,
        qe_ecutwfc=args.qe_ecutwfc,
        qe_ecutrho=args.qe_ecutrho,
        rng_seed=args.rng_seed,
    )
    print(f"Wrote outputs to: {output}")
