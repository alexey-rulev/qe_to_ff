# qe_out_reader.py
import re
import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

RY_TO_EV = 13.605693009
BOHR_TO_ANG = 0.529177210903

def _last_match(pattern, text, flags=0):
    m = None
    for m in re.finditer(pattern, text, flags):
        pass
    return m

def _parse_alat_ang(text):
    """Return alat (Å) if available, else None."""
    # celldm(1) =  X
    m = _last_match(r'celldm\(1\)\s*=\s*([0-9.EDed+\-]+)', text)
    if m:
        return float(m.group(1)) * BOHR_TO_ANG
    # lattice parameter (a_0) = X a.u.
    m = _last_match(r'lattice parameter\s*\(a[_\s]*0?\)\s*=\s*([0-9.EDed+\-]+)\s*a\.?u\.?', text)
    if m:
        return float(m.group(1)) * BOHR_TO_ANG
    return None

def _parse_cell(text):
    # 1) CELL_PARAMETERS block (last occurrence)
    m = _last_match(
        r'CELL_PARAMETERS\s*\(\s*(alat|bohr|angstrom)\s*\)\s*\n'
        r'([^\n]*\n[^\n]*\n[^\n]*\n)',
        text, flags=re.IGNORECASE)
    if m:
        unit = m.group(1).lower()
        rows = [list(map(float, ln.split()[:3])) for ln in m.group(2).strip().splitlines()]
        cell = np.array(rows, float)
        if unit == 'alat':
            alat = _parse_alat_ang(text)
            if alat is None:
                raise ValueError("CELL_PARAMETERS(alat) given but alat not found.")
            cell = cell * alat
        elif unit == 'bohr':
            cell = cell * BOHR_TO_ANG
        elif unit == 'angstrom':
            pass
        return cell

    # 2) "crystal axes" block (numbers are in units of alat)
    ax = re.findall(
        r'a\(\s*[123]\s*\)\s*=\s*\(\s*([0-9eE+\-\.]+)\s+([0-9eE+\-\.]+)\s+([0-9eE+\-\.]+)\s*\)',
        text)
    if len(ax) >= 3:
        alat = _parse_alat_ang(text)
        if alat is None:
            raise ValueError("Found crystal axes but could not find alat (celldm(1) or a0).")
        cell = alat * np.array(ax[:3], float)
        return cell

    raise ValueError("Could not parse CELL (CELL_PARAMETERS or crystal axes).")

def _parse_positions(text, cell):
    # Prefer ATOMIC_POSITIONS blocks (last occurrence)
    m = _last_match(
        r'ATOMIC_POSITIONS\s*\(\s*(crystal|alat|bohr|angstrom)\s*\)\s*\n'
        r'((?:[A-Za-z][A-Za-z0-9_]*\s+[^\n]*\n)+)',
        text, flags=re.IGNORECASE)
    if m:
        unit = m.group(1).lower()
        lines = [ln.strip() for ln in m.group(2).strip().splitlines()]
        symbols, coords = [], []
        for ln in lines:
            parts = ln.split()
            sym = parts[0]
            x, y, z = map(float, parts[1:4])
            symbols.append(sym)
            coords.append([x, y, z])
        R = np.array(coords, float)
        if unit == 'crystal':
            # fractional to cartesian Å
            R = np.dot(R, cell)
        elif unit == 'alat':
            alat = _parse_alat_ang(text)
            if alat is None:
                raise ValueError("ATOMIC_POSITIONS(alat) but alat not found.")
            R = R * alat
        elif unit == 'bohr':
            R = R * BOHR_TO_ANG
        elif unit == 'angstrom':
            pass
        return symbols, R

    # Fallback: "site n. atom positions (alat units) ... tau(i) = ( x y z )"
    # We take the *last* such block by scanning all tau() lines and keeping the last contiguous group.
    tau_iter = list(re.finditer(
        r'^\s*\d+\s+([A-Za-z][A-Za-z0-9_]*)\s+tau\(\s*\d+\s*\)\s*=\s*\(\s*'
        r'([0-9eE+\-\.]+)\s+([0-9eE+\-\.]+)\s+([0-9eE+\-\.]+)\s*\)',
        text, flags=re.IGNORECASE | re.MULTILINE))
    if tau_iter:
        # keep only from the last "site n. atom positions" header onwards if present
        # but generally, taking the last N contiguous matches yields the final geometry
        # Collect trailing block with consecutive atom indices
        # Simpler: collect symbols/coords from the last N unique increasing indices
        # Here, assume the last N matches constitute the last geometry
        # (QE prints one per ionic step).
        # We detect N by the highest 'tau()' index in the tail.
        # Get the last block length by scanning backwards until indices reset to 1.
        # First, refind with the index captured:
        tau_iter2 = list(re.finditer(
            r'^\s*(\d+)\s+([A-Za-z][A-Za-z0-9_]*)\s+tau\(\s*(\d+)\s*\)\s*=\s*\(\s*'
            r'([0-9eE+\-\.]+)\s+([0-9eE+\-\.]+)\s+([0-9eE+\-\.]+)\s*\)',
            text, flags=re.IGNORECASE | re.MULTILINE))
        n = len(tau_iter2)
        # find start of last block
        start = n - 1
        last_idx = int(tau_iter2[start].group(3))
        for i in range(n - 2, -1, -1):
            idx = int(tau_iter2[i].group(3))
            if idx > last_idx:  # still same block
                start = i
                last_idx = idx
            elif idx == 1:  # likely the start of a block
                start = i
                break
        recs = tau_iter2[start:]
        symbols = [m2.group(2) for m2 in recs]
        R_alat = np.array([[float(m2.group(4)), float(m2.group(5)), float(m2.group(6))] for m2 in recs], float)
        alat = _parse_alat_ang(text)
        if alat is None:
            raise ValueError("Positions in alat units but alat not found.")
        R = R_alat * alat  # Cartesian in Å (QE states 'positions (alat units)' are Cartesian)
        return symbols, R

    raise ValueError("Could not parse atomic positions (ATOMIC_POSITIONS or tau()).")

def _parse_forces(text, nat):
    """
    Returns forces in eV/Å, shape (nat,3).
    Looks for the *last* 'Forces acting on atoms (...)' block.
    """
    m = _last_match(
        r'Forces\s+acting\s+on\s+atoms\s*\(([^)]+)\)\s*:\s*\n'
        r'((?:.*force\s*=\s*[^\n]*\n)+)',
        text, flags=re.IGNORECASE)
    if not m:
        return None
    unit = m.group(1).strip().lower()
    lines = [ln for ln in m.group(2).splitlines() if 'force' in ln]
    F = []
    for ln in lines:
        # ... force =   fx   fy   fz
        mm = re.search(r'force\s*=\s*([\-0-9eE\.]+)\s+([\-0-9eE\.]+)\s+([\-0-9eE\.]+)', ln)
        if mm:
            F.append([float(mm.group(1)), float(mm.group(2)), float(mm.group(3))])
    if len(F) != nat:
        # Sometimes QE prints extra lines (e.g., warnings). Keep last nat entries if possible.
        F = F[-nat:]
    F = np.array(F, float)
    # Unit handling: common is 'Ry/au' or 'Ry/Bohr'
    if 'ry' in unit and ('au' in unit or 'bohr' in unit):
        F = F * (RY_TO_EV / BOHR_TO_ANG)
    elif ('ev' in unit and ('a' in unit or 'ang' in unit)):  # eV/Angstrom
        pass
    else:
        # Conservative fallback: assume Ry/Bohr
        F = F * (RY_TO_EV / BOHR_TO_ANG)
    return F

def _parse_total_energy(text):
    # The canonical line is: "!    total energy              =   XXXXX Ry"
    m = _last_match(r'!\s*total\s+energy\s*=\s*([\-0-9eE\.]+)\s*(Ry|eV)', text, flags=re.IGNORECASE)
    if not m:
        # fallback: sometimes printed without '!'
        m = _last_match(r'\btotal\s+energy\s*=\s*([\-0-9eE\.]+)\s*(Ry|eV)', text, flags=re.IGNORECASE)
    if not m:
        return None
    val = float(m.group(1))
    unit = m.group(2).lower()
    if unit == 'ry':
        return val * RY_TO_EV
    return val

def read_qe_out_to_atoms(path):
    """
    Read a Quantum ESPRESSO pw.x .out file and return an ASE Atoms with:
      - cell, pbc=True
      - positions (Å)
      - SinglePointCalculator attached with results:
          energy (eV), forces (eV/Å) if available
    """
    with open(path, 'r', errors='ignore') as f:
        text = f.read()

    # Cell
    cell = _parse_cell(text)

    # Positions
    symbols, positions = _parse_positions(text, cell)
    nat = len(symbols)

    # Forces (optional)
    forces = _parse_forces(text, nat)

    # Energy (optional)
    energy = _parse_total_energy(text)

    atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)

    results = {}
    if energy is not None:
        results['energy'] = float(energy)
        results['free_energy'] = float(energy)  # ASE expects one; set equal if no separate F given
    if forces is not None and len(forces) == nat:
        results['forces'] = forces

    if results:
        calc = SinglePointCalculator(atoms, **results)
        atoms.calc = calc

    return atoms

# --- Example usage ---
# from qe_out_reader import read_qe_out_to_atoms
# ats = read_qe_out_to_atoms('sc-001.out')
# print(ats)
# print("E (eV):", ats.get_potential_energy())
# print("F (eV/Å):\n", ats.get_forces())
# from ase.io import write
# write('structure.cif', ats)
