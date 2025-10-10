
import argparse
from .postprocess import collect_and_compare

def main():
    p = argparse.ArgumentParser(description="Combine QE .out files into EXTXYZ and compare energies with MatterSim.")
    p.add_argument("--dirs", nargs="+", required=True, help="Folders/files to scan for *.out")
    p.add_argument("--out-extxyz", required=True, help="Output EXTXYZ path")
    p.add_argument("--plot", default=None, help="Optional PNG for MatterSim vs QE energy")
    p.add_argument("--mattersim-ckpt", default=None, help="MatterSim checkpoint (optional)")
    p.add_argument("--device", default="cuda", help="'cuda' or 'cpu'")
    p.add_argument("--no-forces", action="store_true", help="Also store MatterSim forces")
    args = p.parse_args()

    res = collect_and_compare(
        search_dirs=args.dirs,
        out_extxyz=args.out_extxyz,
        mattersim_ckpt=args.mattersim_ckpt,
        device=args.device,
        include_forces=not args.no_forces,
        plot_path=args.plot,
    )
    print(f"Wrote {res['n']} frames to {res['extxyz']}")
    if res.get("plot"):
        print(f"Saved plot to {res['plot']}")
