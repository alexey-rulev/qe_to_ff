#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create QE inputs from a template + per-config geometry, then submit each as a SLURM job.

- Template supplies everything (CONTROL/SYSTEM/ELECTRONS/IONS/CELL, ATOMIC_SPECIES, K_POINTS, HUBBARD, etc.)
- Geometry (*.in from your generator) supplies only CELL_PARAMETERS and ATOMIC_POSITIONS.
- nat/ntyp in &SYSTEM are replaced based on the ATOMIC_POSITIONS taken from each geometry file.
- Creates submit scripts and sbatch them (like your example).
- Stdlib only (argparse, glob, os, re, subprocess, time, pathlib).

Usage example:
  python submit_qe_configs.py \
    --template opt.in \
    --inputs "random/qe/*.in" "lattice_scan/qe/*.in" "vacancies/qe/*.in" \
    --out-dir qe_submit \
    --job-name-prefix basno3 \
    --partition compute --nodes 1 --ntasks-per-node 64 --cpus-per-task 1 \
    --launcher prun --qe-exec pw.x --time "12:00:00" --submit

If you want to use current directory *.in files (except the template), just:
  python submit_qe_configs.py --template opt.in --inputs "*.in" --submit
"""

import argparse
import glob
import os
import re
import subprocess
import time
from pathlib import Path
from typing import List, Tuple, Dict


# ---------------------------- small helpers ----------------------------

SECTION_START_RE = re.compile(r'^\s*&([A-Z]+)\s*$')  # &SYSTEM
SECTION_END_RE   = re.compile(r'^\s*/\s*$')          # /
CELL_HEADER_RE   = re.compile(r'^\s*CELL_PARAMETERS\b', re.IGNORECASE)
ATPOS_HEADER_RE  = re.compile(r'^\s*ATOMIC_POSITIONS\b', re.IGNORECASE)
KPTS_HEADER_RE   = re.compile(r'^\s*K_POINTS\b', re.IGNORECASE)
HUBBARD_HEADER_RE= re.compile(r'^\s*HUBBARD\b', re.IGNORECASE)

CAPS_HEADERS = (
    "ATOMIC_SPECIES", "CELL_PARAMETERS", "ATOMIC_POSITIONS", "K_POINTS", "HUBBARD"
)

def _read_text(path: Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return f.readlines()

def _write_text(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)

def _strip_comment(line: str) -> str:
    # crude removal of inline comments after ! or # (QE tolerates ! comments)
    return re.split(r'[#!]', line, maxsplit=1)[0].rstrip("\n")


# ---------------------------- extractors ----------------------------

def extract_qe_section(lines: List[str], name: str) -> Tuple[int, int, List[str]]:
    """
    Extract an &NAME ... / block (inclusive). Returns (start_idx, end_idx_inclusive, block_lines).
    Raises ValueError if not found.
    """
    name = name.upper()
    start = -1
    for i, line in enumerate(lines):
        m = SECTION_START_RE.match(line)
        if m and m.group(1).upper() == name:
            start = i
            break
    if start < 0:
        raise ValueError(f"&{name} section not found")

    for j in range(start + 1, len(lines)):
        if SECTION_END_RE.match(lines[j]):
            end = j
            return start, end, lines[start:end + 1]
    raise ValueError(f"&{name} section does not terminate with '/'")

def extract_block_with_header(lines: List[str], header_re: re.Pattern) -> Tuple[int, int, List[str]]:
    """
    Extract a header block like 'CELL_PARAMETERS...' or 'ATOMIC_POSITIONS...'.
    Continues until a blank line or a new known header or EOF.
    Returns (start, end_inclusive, block_lines). Raises ValueError if not found.
    """
    start = -1
    for i, line in enumerate(lines):
        if header_re.match(line):
            start = i
            break
    if start < 0:
        raise ValueError("Header not found: " + header_re.pattern)

    def is_new_header(s: str) -> bool:
        s0 = s.strip().split()[0] if s.strip() else ""
        if s0.upper().startswith("&") or SECTION_END_RE.match(s):
            return True
        for h in CAPS_HEADERS:
            if s.strip().upper().startswith(h):
                return True
        return False

    block = [lines[start]]
    for j in range(start + 1, len(lines)):
        if lines[j].strip() == "":
            block.append(lines[j])
            return start, j, block
        if is_new_header(lines[j]):
            # stop BEFORE this header
            return start, j - 1, block
        block.append(lines[j])
    return start, len(lines) - 1, block

def extract_atomic_positions(lines: List[str]) -> Tuple[List[str], str]:
    """
    Returns (atom_lines, header_line), where atom_lines are the lines with species+coords under ATOMIC_POSITIONS.
    """
    _, _, block = extract_block_with_header(lines, ATPOS_HEADER_RE)
    header = block[0].rstrip("\n")
    atoms = []
    for ln in block[1:]:
        if not ln.strip():
            break
        atoms.append(ln.rstrip("\n"))
    return atoms, header

def count_nat_ntyp(atom_lines: List[str]) -> Tuple[int, int]:
    species = []
    for ln in atom_lines:
        s = _strip_comment(ln).strip()
        if not s:
            continue
        parts = s.split()
        if not parts:
            continue
        species.append(parts[0])
    nat = len(species)
    ntyp = len(set(species))
    return nat, ntyp


# ---------------------------- modifiers ----------------------------

def replace_nat_ntyp_in_system(system_block: List[str], nat: int, ntyp: int) -> List[str]:
    """
    Replace nat and ntyp lines inside &SYSTEM ... / block. Keeps formatting where possible.
    Adds the key if missing.
    """
    out = []
    seen_nat = False
    seen_ntyp = False
    for ln in system_block:
        if re.search(r'\bnat\b', ln) and "=" in ln:
            out.append(re.sub(r'(?i)\bnat\b\s*=\s*[^,/\n]+', f"nat = {nat}", ln))
            seen_nat = True
        elif re.search(r'\bntyp\b', ln) and "=" in ln:
            out.append(re.sub(r'(?i)\bntyp\b\s*=\s*[^,/\n]+', f"ntyp = {ntyp}", ln))
            seen_ntyp = True
        else:
            out.append(ln)

    # insert missing keys before closing '/'
    if not seen_nat or not seen_ntyp:
        inserted = []
        for ln in out:
            if ln.strip() == "/" and (not seen_nat or not seen_ntyp):
                if not seen_nat:
                    inserted.append(f"   nat             =  {nat}\n")
                    seen_nat = True
                if not seen_ntyp:
                    inserted.append(f"   ntyp            =  {ntyp}\n")
                    seen_ntyp = True
            inserted.append(ln)
        out = inserted
    return out

def splice_template_with_geometry(
    template_lines: List[str],
    geom_lines: List[str],
    override_prefix: str = None
) -> List[str]:
    """
    Build final .in:
      &CONTROL (from template; optionally override prefix)
      &SYSTEM   (from template, but nat/ntyp replaced)
      &ELECTRONS (from template)
      &IONS     (from template, if present)
      &CELL     (from template)
      ATOMIC_SPECIES (from template)
      CELL_PARAMETERS (from geometry)
      ATOMIC_POSITIONS (from geometry)
      K_POINTS (from template, if present)
      HUBBARD (from template, if present)
    """
    out: List[str] = []

    # --- CONTROL
    c0, c1, control = extract_qe_section(template_lines, "CONTROL")
    if override_prefix is not None:
        new_control = []
        for ln in control:
            if re.search(r"\bprefix\b", ln) and "=" in ln:
                new_control.append(re.sub(r"(?i)\bprefix\b\s*=\s*[^,\n]+", f"prefix = '{override_prefix}'", ln))
            else:
                new_control.append(ln)
        control = new_control
    out.extend(control)
    out.append("\n")

    # --- SYSTEM (replace nat, ntyp)
    s0, s1, system = extract_qe_section(template_lines, "SYSTEM")
    # Get nat/ntyp from geometry
    atom_lines, atpos_header = extract_atomic_positions(geom_lines)
    nat, ntyp = count_nat_ntyp(atom_lines)
    system = replace_nat_ntyp_in_system(system, nat=nat, ntyp=ntyp)
    out.extend(system)
    out.append("\n")

    # --- ELECTRONS (if exists)
    try:
        e0, e1, electrons = extract_qe_section(template_lines, "ELECTRONS")
        out.extend(electrons); out.append("\n")
    except ValueError:
        pass

    # --- IONS (optional)
    try:
        i0, i1, ions = extract_qe_section(template_lines, "IONS")
        out.extend(ions); out.append("\n")
    except ValueError:
        pass

    # --- CELL (optional)
    try:
        cl0, cl1, cell = extract_qe_section(template_lines, "CELL")
        out.extend(cell); out.append("\n")
    except ValueError:
        pass

    # --- ATOMIC_SPECIES (from template)
    try:
        as0, as1, atomic_species = extract_block_with_header(template_lines, re.compile(r'^\s*ATOMIC_SPECIES\b', re.IGNORECASE))
        out.extend(atomic_species); out.append("\n")
    except ValueError:
        # If not present, user said they'll place PPs alongside; QE can still run if PPs are in pseudopotential dir with default names.
        pass

    # --- CELL_PARAMETERS (from geometry)
    cp0, cp1, cell_params = extract_block_with_header(geom_lines, CELL_HEADER_RE)
    out.extend(cell_params); out.append("\n")

    # --- ATOMIC_POSITIONS (from geometry)
    ap0, ap1, atpos_block = extract_block_with_header(geom_lines, ATPOS_HEADER_RE)
    out.extend(atpos_block); out.append("\n")

    # --- K_POINTS (from template)
    try:
        k0, k1, kpts = extract_block_with_header(template_lines, KPTS_HEADER_RE)
        out.extend(kpts); out.append("\n")
    except ValueError:
        pass

    # --- HUBBARD (from template; may be multi-line)
    # We'll copy from first 'HUBBARD' line to EOF.
    for i, ln in enumerate(template_lines):
        if HUBBARD_HEADER_RE.match(ln):
            out.extend(template_lines[i:])
            if not out[-1].endswith("\n"):
                out.append("\n")
            break

    return out


# ---------------------------- slurm script ----------------------------

SLURM_TEMPLATE = """#!/bin/bash -l
#SBATCH --job-name="{job}"
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={ntpn}
#SBATCH --cpus-per-task={cpt}
#SBATCH --ntasks-per-core={ntpc}
#SBATCH --partition={part}
{time_line}{account_line}
module load {module_qe}

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

{launcher} {qe_exec} < {infile} > {outfile}
"""

def write_slurm_submit(
    path: Path,
    job: str,
    nodes: int,
    ntpn: int,
    cpt: int,
    ntpc: int,
    part: str,
    module_qe: str,
    launcher: str,
    qe_exec: str,
    infile: str,
    outfile: str,
    time_str: str = None,
    account: str = None,
) -> None:
    time_line = f"#SBATCH --time={time_str}\n" if time_str else ""
    account_line = f"#SBATCH --account={account}\n" if account else ""
    txt = SLURM_TEMPLATE.format(
        job=job, nodes=nodes, ntpn=ntpn, cpt=cpt, ntpc=ntpc, part=part,
        time_line=time_line, account_line=account_line, module_qe="qe",
        launcher=launcher, qe_exec=qe_exec, infile=infile, outfile=outfile
    )
    _write_text(path, [txt])


# ---------------------------- main ----------------------------

def main():
    ap = argparse.ArgumentParser(description="Assemble QE inputs from a template and submit to SLURM.")
    ap.add_argument("--template", required=True, help="Template QE input (e.g., opt.in)")
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="Glob(s) for geometry .in files (CELL_PARAMETERS + ATOMIC_POSITIONS).")
    ap.add_argument("--out-dir", default="qe_submit", help="Where to write final inputs and submit scripts.")
    ap.add_argument("--job-name-prefix", default="conf", help="SLURM job-name prefix (suffix will be index).")
    ap.add_argument("--override-prefix", default=None,
                    help="If set, overrides 'prefix' inside &CONTROL with this value + index (e.g., conf_001).")
    ap.add_argument("--launcher", default="prun", choices=["prun", "srun", "mpirun"],
                    help="MPI launcher in submit script.")
    ap.add_argument("--qe-exec", default="pw.x", help="QE executable (pw.x by default).")
    ap.add_argument("--partition", default="compute", help="SLURM partition.")
    ap.add_argument("--nodes", type=int, default=1)
    ap.add_argument("--ntasks-per-node", type=int, default=64)
    ap.add_argument("--cpus-per-task", type=int, default=1)
    ap.add_argument("--ntasks-per-core", type=int, default=1)
    ap.add_argument("--time", default=None, help="SLURM time limit, e.g., 12:00:00")
    ap.add_argument("--account", default=None, help="SLURM account")
    ap.add_argument("--sleep", type=float, default=2.0, help="Delay seconds between sbatch submissions.")
    ap.add_argument("--no-submit", action="store_true", help="Create files but do not call sbatch.")
    args = ap.parse_args()

    template_path = Path(args.template).resolve()
    if not template_path.is_file():
        raise SystemExit(f"Template not found: {template_path}")

    # Collect geometry files from globs, keep stable order
    geom_files: List[Path] = []
    for g in args.inputs:
        geom_files.extend(Path(p).resolve() for p in glob.glob(g))
    # Drop the template itself if matched by a glob
    geom_files = [p for p in sorted(set(geom_files)) if p != template_path]
    if not geom_files:
        raise SystemExit("No geometry .in files matched the provided globs.")

    out_root = Path(args.out_dir).resolve()
    out_in_dir = out_root / "in"
    out_slurm_dir = out_root / "slurm"
    out_root.mkdir(parents=True, exist_ok=True)
    out_in_dir.mkdir(parents=True, exist_ok=True)
    out_slurm_dir.mkdir(parents=True, exist_ok=True)

    tmpl = _read_text(template_path)

    # Build each job
    jobs: List[Tuple[Path, Path]] = []  # (input_file, submit_script)
    for idx, geom_path in enumerate(geom_files, start=1):
        geom = _read_text(geom_path)

        # output names
        tag = f"{idx:03d}"
        out_in = out_in_dir / f"conf_{tag}.in"
        out_out = out_in_dir / f"conf_{tag}.out"
        out_sh = out_slurm_dir / f"submit-{tag}.sh"

        # Always generate unique prefix based on filename or user override
        if args.override_prefix:
            prefix_override = f"{args.override_prefix}_{tag}"
        else:
            # Use the output filename stem as unique prefix
            prefix_override = out_in.stem

        # splice and write .in
        merged = splice_template_with_geometry(tmpl, geom, override_prefix=prefix_override)
        _write_text(out_in, merged)

        # write submit script
        jobname = f"{args.job-name-prefix}-{tag}" if hasattr(args, "job-name-prefix") else f"{args.job_name_prefix}-{tag}"
        # Python argparse converts dashes to underscores; correct that:
        jobname = f"{getattr(args, 'job_name_prefix').replace('_', '-')}-{tag}"
        write_slurm_submit(
            out_sh,
            job=jobname,
            nodes=args.nodes,
            ntpn=args.ntasks_per_node,
            cpt=args.cpus_per_task,
            ntpc=args.ntasks_per_core,
            part=args.partition,
            module_qe="qe",
            launcher=args.launcher,
            qe_exec=args.qe_exec,
            infile=out_in.name,
            outfile=out_out.name,
            time_str=args.time,
            account=args.account,
        )
        jobs.append((out_in, out_sh))

    # Submit from out_in_dir (so relative infile/outfile paths resolve)
    if not args.no_submit:
        for _, sh in jobs:
            # Submit inside in/ dir so "< conf_XXX.in > conf_XXX.out" works
            subprocess.run(["sbatch", str(sh.resolve())], cwd=str(out_in_dir), check=False)
            time.sleep(args.sleep)

    print(f"Prepared {len(jobs)} jobs in: {out_root}")
    if args.no_submit:
        print("NOTE: --no-submit set; jobs were not submitted.")
    else:
        print("Submitted with sbatch.")

if __name__ == "__main__":
    main()
