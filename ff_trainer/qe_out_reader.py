import re
import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

RY_TO_EV = 13.605693009
BOHR_TO_ANG = 0.529177210903
KBAR_TO_EV_A3 = 0.0006241509074460765  # 1 kbar = 0.0006241509074 eV/Å^3
RY_PER_BOHR3_TO_EV_A3 = RY_TO_EV / (BOHR_TO_ANG ** 3)

def _last_match(pattern, text, flags=0):
    m = None
    for m in re.finditer(pattern, text, flags):
        pass
    return m

def _parse_alat_ang(text):
    """Return alat (Å) if available, else None."""
    m = _last_match(r'celldm\(1\)\s*=\s*([0-9.EDed+\-]+)', text)
    if m:
        return float(m.group(1)) * BOHR_TO_ANG
    m = _last_match(r'lattice parameter\s*\(a[_\s]*0?\)\s*=\s*([0-9.EDed+\-]+)\s*a\.?u\.?', text)
    if m:
        return float(m.group(1)) * BOHR_TO_ANG
    return None

def _parse_cell(text):
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

    tau_iter = list(re.finditer(
        r'^\s*(\d+)\s+([A-Za-z][A-Za-z0-9_]*)\s+tau\(\s*(\d+)\s*\)\s*=\s*\(\s*'
        r'([0-9eE+\-\.]+)\s+([0-9eE+\-\.]+)\s+([0-9eE+\-\.]+)\s*\)',
        text, flags=re.IGNORECASE | re.MULTILINE))
    if tau_iter:
        n = len(tau_iter)
        start = n - 1
        last_idx = int(tau_iter[start].group(3))
        for i in range(n - 2, -1, -1):
            idx = int(tau_iter[i].group(3))
            if idx > last_idx:
                start = i
                last_idx = idx
            elif idx == 1:
                start = i
                break
        recs = tau_iter[start:]
        symbols = [m2.group(2) for m2 in recs]
        R_alat = np.array([[float(m2.group(4)), float(m2.group(5)), float(m2.group(6))] for m2 in recs], float)
        alat = _parse_alat_ang(text)
        if alat is None:
            raise ValueError("Positions in alat units but alat not found.")
        R = R_alat * alat
        return symbols, R

    raise ValueError("Could not parse atomic positions (ATOMIC_POSITIONS or tau()).")

def _parse_forces(text, nat):
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
        mm = re.search(r'force\s*=\s*([\-0-9eE\.]+)\s+([\-0-9eE\.]+)\s+([\-0-9eE\.]+)', ln)
        if mm:
            F.append([float(mm.group(1)), float(mm.group(2)), float(mm.group(3))])
    if len(F) != nat:
        F = F[-nat:]
    F = np.array(F, float)
    if 'ry' in unit and ('au' in unit or 'bohr' in unit):
        F = F * (RY_TO_EV / BOHR_TO_ANG)
    elif ('ev' in unit and ('a' in unit or 'ang' in unit)):
        pass
    else:
        F = F * (RY_TO_EV / BOHR_TO_ANG)
    return F

def _parse_total_energy(text):
    m = _last_match(r'!\s*total\s+energy\s*=\s*([\-0-9eE\.]+)\s*(Ry|eV)', text, flags=re.IGNORECASE)
    if not m:
        m = _last_match(r'\btotal\s+energy\s*=\s*([\-0-9eE\.]+)\s*(Ry|eV)', text, flags=re.IGNORECASE)
    if not m:
        return None
    val = float(m.group(1))
    unit = m.group(2).lower()
    if unit == 'ry':
        return val * RY_TO_EV
    return val

def _parse_stress(text):
    """
    Return 3x3 stress tensor in eV/Å^3 from the last 'total   stress (...)' block.
    QE often prints both units in the header (Ry/bohr**3) (kbar) and two 3x3 matrices
    side-by-side; we take the first three numbers on each of the three lines and
    convert using the unit captured from the header.
    """
    m = _last_match(r'total\s+stress\s*\(([^)]+)\)', text, flags=re.IGNORECASE)
    if not m:
        return None
    unit = m.group(1).strip().lower()
    # Get the next 3 lines with numbers
    pos = m.end()
    tail = text[pos:].splitlines()
    rows = []
    for ln in tail:
        # capture all numeric fields on the line
        nums = re.findall(r'[-+]?\d*\.?\d+(?:[eEdD][-\+]?\d+)?', ln)
        if len(nums) >= 3:
            rows.append([float(nums[0]), float(nums[1]), float(nums[2])])
            if len(rows) == 3:
                break
        # skip blank / non-numeric lines
    if len(rows) != 3:
        return None
    S = np.array(rows, float)

    u = unit.replace("**", "^")
    if ('ry' in u) and ('bohr' in u or 'au' in u):
        S = S * RY_PER_BOHR3_TO_EV_A3
    elif 'kbar' in u:
        S = S * KBAR_TO_EV_A3
    elif 'ev' in u and ('a' in u or 'ang' in u):
        # already eV/Å^3
        pass
    else:
        # conservative fallback
        S = S * RY_PER_BOHR3_TO_EV_A3
    return S

def read_qe_out_to_atoms(path):
    """
    Read a Quantum ESPRESSO pw.x .out file and return an ASE Atoms with:
      - cell, pbc=True
      - positions (Å)
      - SinglePointCalculator attached with results:
          energy (eV), forces (eV/Å), stress (eV/Å^3) if available
    """
    with open(path, 'r', errors='ignore') as f:
        text = f.read()

    cell = _parse_cell(text)
    symbols, positions = _parse_positions(text, cell)
    nat = len(symbols)

    forces = _parse_forces(text, nat)
    energy = _parse_total_energy(text)
    stress = _parse_stress(text)

    atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)

    results = {}
    if energy is not None:
        results['energy'] = float(energy)
        results['free_energy'] = float(energy)
    if forces is not None and len(forces) == nat:
        results['forces'] = forces
    if stress is not None:
        # ASE expects stress in eV/Å^3 (3x3); calculators may use either sign convention;
        # we pass QE's sign as-is to keep raw parity with QE output.
        results['stress'] = stress

    if results:
        calc = SinglePointCalculator(atoms, **results)
        atoms.calc = calc

    return atoms
