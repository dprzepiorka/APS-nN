"""
Metaheuristic Optimization for DIgSILENT PowerFactory
- Funkcja celu złożona z 5 składowych + kary
- Zapis komponentów i historii do Excela
- Obsługa wielu algorytmów optymalizacji (PSO, CEO, ... COO)
Author: integrated/modified for user by ChatGPT (Copilot Space)
Original repository: dprzepiorka/ELVTF (adapted)
"""
import sys
import os
import re
import time
import math
import random
import shutil
import subprocess
import tempfile
import traceback
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import special as sps

# Import optimizers (assumed present in repo)
from CEO import CEO
from PSO import PSO
from PO import PO
from DOA import DOA
from RTO import RTO
from SPO import SPO
from BPB import BPB
from KLA import KLA
from KEO import KEO
from DOE import DOE
from OSA import OSA
from ECO import ECO
from GOA import GOA
from COO import COO

# PowerFactory python path - adjust if necessary for your installation
sys.path.append(r"C:\Program Files\DIgSILENT\PowerFactory 2024 SP5A\Python\3.12")
try:
    import powerfactory
except Exception:
    powerfactory = None
    print("Warning: powerfactory module not available (script still importable for offline tests).")

from scipy.optimize import differential_evolution, dual_annealing, minimize

# -------------------------
# CONFIG (ready-to-run defaults; change only if you really must)
# -------------------------
EXCEL_FILE = r"N:\ksiz\STUDIA_DOKTORANCKIE\PRZEPIORKA\Modele sieci\Niskie napiecie\APS\Optymalizacja\dane.xlsx"
PROJECT_NAME = "ELVTF3x1F"
OUT_FILE = r"N:\ksiz\STUDIA_DOKTORANCKIE\PRZEPIORKA\Modele sieci\Niskie napiecie\APS\Optymalizacja\wyniki_Opt.xlsx"
START_FILE = r"N:\ksiz\STUDIA_DOKTORANCKIE\PRZEPIORKA\Modele sieci\Niskie napiecie\APS\Optymalizacja\wyniki_Pocz.xlsx"
USER = "minik"

METHOD = "ECO"  # "PSO", "CEO", "PO", "DOA", "RTO", "SPO- długi", "BPB", "KLA", "KEO", "DOE", "OSA", "ECO", "GOA", "COO"
OBJECTIVE = "VoltageTarget"  # "LossP" or "VoltageUnbalance" or "VoltageTarget"

N_ITER = 1000
N_PARTICLES = 100
N = 1
W = 0.7
C1 = 1.5
C2 = 1.5
PENALTY = 0 # 1e6
# Multiplier applied when loadflow clearly failed (very large penalty)
LARGE_PENALTY_MULTIPLIER = 0 # 1e4

RANDOM_SEED = 42

# Small delay after each PF run (avoid overloading PowerFactory)
EVAL_DELAY = 0.01

# How often to print periodic progress (every N evaluations)
PRINT_EVERY = 50

# ---- Symmetric phase mode ----
# When True, devices grouped by base name (e.g. ES110, PV5) will use the same
# optimized value for all three phases (F1=F2=F3). Reduces search-space dimension
# significantly without any Excel changes.
SYMMETRIC_PHASES = True

# When True, storage optimizer uses one P and one Q variable for all 3 phases
# (P1=P2=P3=P_sym,  Q1=Q2=Q3=Q_sym). Ignored when no StorageCandidates sheet.
SYMMETRIC_STORAGE = True

# STORAGE defaults (per phase)
STORAGE_P_MIN = -50.0
STORAGE_P_MAX = 50.0
STORAGE_Q_MIN = -0.0001
STORAGE_Q_MAX = 0.0001

# -------------------------
# Surface scan (2D grid) configuration
# -------------------------
# SURFACE_ENABLED=True triggers a 2D scan of the objective function around optimum x*
# after the optimisation run finishes.
#
# HOW TO USE:
#   1. Set SURFACE_ENABLED = True
#   2. Adjust GRID_NP / GRID_NQ (resolution). Warning: large values → GRID_NP*GRID_NQ LF runs.
#   3. Set DP_MIN/DP_MAX [kW] – range of delta for total storage active power    ΔP_stor_sum
#   4. Set DQ_MIN/DQ_MAX [kvar] – range of delta for total reactive power         ΔQ_total_sum
#      (reactive power of all sources ElmSym/ElmPvsys/ElmGenstat + storage Q)
#   5. Run the script. After optimisation the scan executes automatically.
#
# OUTPUT FILES (written to same folder as OUT_FILE):
#   - OUT_SURFACE_FILE  (.xlsx)  – sheets: Axis_dP, Axis_dQ, Surface_J, Surface_meta
#   - surface_scan.npz           – NumPy archive: dP, dQ, J, meta scalars
#   - Surface_3D.png             – 3D surface plot
#   - Surface_Heatmap.png        – 2D contourf heatmap
#
SURFACE_ENABLED = True
RUN_OPTIMIZATION = False   # False = tylko płaszczyzna z zapisanego optimum
GRID_NP = 5            # columns – ΔP axis
GRID_NQ = 5            # rows    – ΔQ axis
DP_MIN  = -20.0         # [kW]  lower bound of ΔP_stor_sum
DP_MAX  =  20.0         # [kW]  upper bound of ΔP_stor_sum
DQ_MIN  = -0.0001       # [kvar] lower bound of ΔQ_total_sum
DQ_MAX  =  0.0001       # [kvar] upper bound of ΔQ_total_sum
EPS_WEIGHT = 1e-6       # epsilon added to |x*| when computing redistribution weights
OUT_SURFACE_FILE = r"N:\ksiz\STUDIA_DOKTORANCKIE\PRZEPIORKA\Modele sieci\Niskie napiecie\APS\Optymalizacja\surface_scan.xlsx"
CHECKPOINT_FILE = None #r"N:\ksiz\STUDIA_DOKTORANCKIE\PRZEPIORKA\Modele sieci\Niskie napiecie\APS\Optymalizacja\eco_checkpoint.npz"  # np. r"N:\...\eco_checkpoint.npz"; None = wybór automatyczny z METHOD
BEST_SOLUTION_FILE = r"N:\ksiz\STUDIA_DOKTORANCKIE\PRZEPIORKA\Modele sieci\Niskie napiecie\APS\Optymalizacja\best_solution.txt"

VOLTAGE_MIN = 0.9
VOLTAGE_MAX = 1.1
LOAD_MAX = 100.0  # [%]

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ----- Weights for objective components (sum to 1)
# Order: [C1_Udev_phases, C2_Iasym_line1002, C3_Ploss_norm, C4_V0_coeff_sum, C5_V2_coeff_sum]

#WEIGHTS = np.array([0.2, 0.2, 0.2, 0.2, 0.2], dtype=float)      #Po równo
WEIGHTS = np.array([0.126308, 0.800897, 0.028356, 0.028608, 0.016131], dtype=float)      #Dobrane

# Line name used for C2 (as requested): "1002"
LINE_NOM_NAME = "1002"

# Global history of component evaluations (list of [C1,C2,C3,C4,C5,penalty,total_obj])
COMPONENT_HISTORY = []

# Global storage candidates list — loaded in main()
STORAGE_CANDIDATES = []

# Global mirror map — built in main() when SYMMETRIC_PHASES=True.
# Key: (primary_name, attr)  Value: list of (mirror_name, pf_class, attr)
MIRROR_MAP = {}

# -------------------------
# Helper functions
# -------------------------
def find_element(app, name, pf_class):
    """
    Try several ways to find an element by name and class in PowerFactory.
    """
    try:
        objs = app.GetCalcRelevantObjects(f"{name}.{pf_class}")
        if objs:
            return objs[0]
    except Exception:
        pass
    try:
        for o in app.GetCalcRelevantObjects(f"*.{pf_class}"):
            if getattr(o, "loc_name", None) == name:
                return o
    except Exception:
        pass
    return None


def read_excel_safe(file_path, sheet_name=None, **kwargs):
    """
    Read Excel robustly in PowerFactory/Windows environments.

    Some setups fail on mapped/network drives with:
      OSError: [Errno 9] Bad file descriptor

    Strategy:
      1. Normal pandas read
            2. Fallback: copy source workbook to a local temp file and read from there
                 using external system copy if Python cannot read the mapped drive directly
    """
    path = os.fspath(file_path)

    try:
        return pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl", **kwargs)
    except Exception as e1:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Nie znaleziono pliku Excel: {path}") from e1

        tmp_path = None
        try:
            fd, tmp_path = tempfile.mkstemp(suffix=".xlsx")
            os.close(fd)

            copied = False

            # 1) Fast Python copy (works on many systems)
            try:
                shutil.copyfile(path, tmp_path)
                copied = True
            except Exception:
                copied = False

            # 2) Fallback to cmd.exe copy (often works better for mapped drives)
            if not copied:
                cmd = f'copy /Y "{path}" "{tmp_path}" >nul'
                res = subprocess.run(["cmd", "/c", cmd], capture_output=True, text=True)
                copied = (res.returncode == 0 and os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0)

            # 3) Fallback to PowerShell Copy-Item
            if not copied:
                ps_cmd = (
                    "Copy-Item -LiteralPath '{src}' -Destination '{dst}' -Force"
                    .format(src=path.replace("'", "''"), dst=tmp_path.replace("'", "''"))
                )
                res = subprocess.run(
                    ["powershell", "-NoProfile", "-Command", ps_cmd],
                    capture_output=True,
                    text=True,
                )
                copied = (res.returncode == 0 and os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0)

            if not copied:
                raise RuntimeError(f"Nie udało się skopiować pliku Excel do pliku tymczasowego: {path}")

            print(f"Uwaga: bezpośredni odczyt Excel nie powiódł się, używam kopii tymczasowej: {sheet_name}")
            return pd.read_excel(tmp_path, sheet_name=sheet_name, engine="openpyxl", **kwargs)
        except Exception as e2:
            raise RuntimeError(
                f"Nie udało się odczytać Excel '{path}' (sheet={sheet_name}). "
                f"Błąd bezpośredni: {e1}; błąd z kopii tymczasowej: {e2}"
            ) from e2
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass


def load_vars(file_path, sheet="Vars"):
    """
    Load variable definitions from Excel sheet "Vars" (columns: name, pf_class, attr, min, max)
    """
    df = read_excel_safe(file_path, sheet_name=sheet)
    df = df.rename(columns={c.strip().lower(): c.strip() for c in df.columns})
    vars_list = []
    for _, row in df.iterrows():
        vars_list.append({
            "name": str(row["name"]).strip(),
            "pf_class": str(row["pf_class"]).strip(),
            "attr": str(row["attr"]).strip(),
            "min": float(row["min"]),
            "max": float(row["max"])
        })
    return vars_list


def load_vars_from_best_solution(file_path):
    """
    Reconstruct vars_def from best_solution.txt when Excel is not accessible.

    Non-storage variables are fixed at the saved optimum (min=max=value), which is
    acceptable in surface-only mode because only storage/Q-redistribution is varied.
    Storage variables receive normal storage bounds so the scan can move them.
    """
    vars_list = []
    in_vars_section = False

    with open(file_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            stripped = line.strip()
            normalized = " ".join(stripped.split())

            if not stripped:
                continue
            if normalized.startswith("Best solution variables:"):
                in_vars_section = True
                continue
            if normalized.startswith("Storage elements attributes:"):
                break
            if not in_vars_section:
                continue
            if normalized.startswith("variable value pf_class"):
                continue

            parts = stripped.split()
            if len(parts) < 3:
                continue

            name = parts[0]
            try:
                value = float(parts[1])
            except Exception:
                continue
            pf_class = parts[2]
            attr = parts[3] if len(parts) >= 4 else ""

            row = {
                "name": name,
                "pf_class": pf_class,
                "attr": attr,
            }

            if name == "storage_node_index":
                row["min"] = value
                row["max"] = value
            elif name.startswith("storage_P"):
                row["min"] = STORAGE_P_MIN
                row["max"] = STORAGE_P_MAX
            elif name.startswith("storage_Q"):
                row["min"] = STORAGE_Q_MIN
                row["max"] = STORAGE_Q_MAX
            else:
                row["min"] = value
                row["max"] = value

            vars_list.append(row)

    if not vars_list:
        raise RuntimeError(f"Nie udało się odtworzyć vars_def z pliku: {file_path}")

    return vars_list


def load_storage_candidates_from_best_solution(file_path):
    """
    Reconstruct a single storage candidate from the 'Storage elements attributes' table
    saved in best_solution.txt.
    """
    phase_map = {}
    in_storage_section = False

    with open(file_path, "r", encoding="utf-8") as f:
        for raw in f:
            stripped = raw.strip()
            normalized = " ".join(stripped.split())
            if not stripped:
                continue
            if normalized.startswith("Storage elements attributes:"):
                in_storage_section = True
                continue
            if not in_storage_section:
                continue
            if normalized.startswith("elem_name phase"):
                continue

            parts = stripped.split()
            if len(parts) < 2:
                continue
            elem_name = parts[0]
            phase = parts[1].upper()
            if phase in ("A", "B", "C"):
                phase_map[phase] = elem_name

    if not phase_map:
        return []

    return [{
        "node": "best_solution_storage",
        "elem_A": phase_map.get("A", ""),
        "elem_B": phase_map.get("B", ""),
        "elem_C": phase_map.get("C", ""),
    }]


_PHASE_RE = re.compile(r'^(.+)(F[123])$', re.IGNORECASE)


def build_symmetric_vars(vars_def):
    """
    Detect variables whose names end in F1/F2/F3 (same base name + attr).
    Keep only the first-encountered phase as the 'primary' (optimization variable).
    All other phases are recorded as 'mirrors' in the returned mirror_map.
    Exact duplicate rows (same name + pf_class + attr) are silently dropped.

    Returns:
        reduced_vars  – vars_def with duplicate-phase rows removed
        mirror_map    – {(primary_name, attr): [(mirror_name, pf_class, attr), ...]}
    """
    seen_keys  = {}   # (base, pf_class, attr)     -> primary var
    seen_exact = set()  # (name, pf_class, attr)   – deduplication guard
    reduced    = []
    mirror_map = {}

    for var in vars_def:
        exact_id = (var["name"], var["pf_class"], var["attr"])

        # skip exact duplicates (same element appears twice in Excel)
        if exact_id in seen_exact:
            continue
        seen_exact.add(exact_id)

        m = _PHASE_RE.match(var["name"])
        if m:
            base = m.group(1)
            key  = (base, var["pf_class"], var["attr"])
            if key not in seen_keys:
                # first phase encountered becomes the primary
                seen_keys[key] = var
                reduced.append(var)
                mirror_map[(var["name"], var["attr"])] = []
            else:
                # subsequent phases → register as mirror of the primary
                primary = seen_keys[key]
                mirror_map[(primary["name"], primary["attr"])].append(
                    (var["name"], var["pf_class"], var["attr"])
                )
        else:
            # non-phase variable — always keep
            reduced.append(var)

    # remove empty mirror lists for cleanliness
    mirror_map = {k: v for k, v in mirror_map.items() if v}
    return reduced, mirror_map


def _apply_mirrors(app, var, value, mirror_map):
    """
    After setting a primary variable, propagate the same value to all its mirrors.
    """
    mirrors = mirror_map.get((var["name"], var["attr"]), [])
    for (mname, mpf_class, mattr) in mirrors:
        elm = find_element(app, mname, mpf_class)
        if elm is not None:
            try:
                elm.SetAttribute(mattr, float(value))
            except Exception:
                pass


def set_single_attribute(app, var, value):
    """
    Set one attribute of an element from vars_def safely.
    """
    elm = find_element(app, var["name"], var["pf_class"])
    if elm is None:
        return False
    try:
        elm.SetAttribute(var["attr"], float(value))
        return True
    except Exception:
        return False


def apply_solution(app, vars_def, x):
    """
    Apply continuous variables (non-storage) from x to PF objects.
    """
    for idx, var in enumerate(vars_def):
        try:
            # Skip storage- related names here (handled separately)
            if var["name"].startswith("storage_"):
                continue
            set_single_attribute(app, var, float(x[idx]))
        except Exception:
            continue


# -------------------------
# Storage helpers (as in original file)
# -------------------------
def load_storage_candidates(file_path, sheet="StorageCandidates"):
    """
    Read storage candidate rows with columns: node, elem_A, elem_B, elem_C
    """
    try:
        df = read_excel_safe(file_path, sheet_name=sheet)
        df = df.fillna("")
        candidates = []
        for _, r in df.iterrows():
            candidates.append({
                "node": str(r.get("node", "")).strip(),
                "elem_A": str(r.get("elem_A", "")).strip(),
                "elem_B": str(r.get("elem_B", "")).strip(),
                "elem_C": str(r.get("elem_C", "")).strip(),
            })
        return candidates
    except Exception as e:
        print(f"Nie udało się wczytać {sheet}: {e}")
        return []


def _set_element_attr_safe(elm, attr, val):
    try:
        elm.SetAttribute(attr, float(val))
        return True
    except Exception:
        return False


def apply_storage_on_node(app, candidate, P1, P2, P3, Q1, Q2, Q3):
    """
    Set per-phase storage values on candidate elements; return prev values dict to restore later.
    """
    prev = {}
    elems = [candidate.get("elem_A"), candidate.get("elem_B"), candidate.get("elem_C")]
    Ps = [P1, P2, P3]
    Qs = [Q1, Q2, Q3]
    for i, ename in enumerate(elems):
        if not ename:
            prev[ename] = None
            continue
        elm = find_element(app, ename, "ElmGenstat") or find_element(app, ename, "ElmSym") or \
              find_element(app, ename, "ElmPvsys") or find_element(app, ename, "ElmLod")
        if not elm:
            prev[ename] = None
            continue
        got = {}
        # try per-phase attributes (loads/phase)
        try:
            got["plinir"] = elm.GetAttribute("plinir")
            got["plinis"] = elm.GetAttribute("plinis")
            got["plinit"] = elm.GetAttribute("plinit")
            got["qlinir"] = elm.GetAttribute("qlinir")
            got["qlinis"] = elm.GetAttribute("qlinis")
            got["qlinit"] = elm.GetAttribute("qlinit")
            if i == 0:
                _set_element_attr_safe(elm, "plinir", Ps[i])
                _set_element_attr_safe(elm, "qlinir", Qs[i])
            elif i == 1:
                _set_element_attr_safe(elm, "plinis", Ps[i])
                _set_element_attr_safe(elm, "qlinis", Qs[i])
            else:
                _set_element_attr_safe(elm, "plinit", Ps[i])
                _set_element_attr_safe(elm, "qlinit", Qs[i])
            prev[ename] = got
            continue
        except Exception:
            pass
        # fallback: sumaryczne attributes
        try:
            got2 = {"pgini": elm.GetAttribute("pgini"), "qgini": elm.GetAttribute("qgini")}
            _set_element_attr_safe(elm, "pgini", Ps[i])
            _set_element_attr_safe(elm, "qgini", Qs[i])
            prev[ename] = got2
        except Exception:
            prev[ename] = None
    return prev


def reset_storage_on_node(app, candidate, prev_values):
    """
    Restore previous attributes saved by apply_storage_on_node.
    """
    elems = [candidate.get("elem_A"), candidate.get("elem_B"), candidate.get("elem_C")]
    for ename in elems:
        if not ename:
            continue
        prev = prev_values.get(ename, None)
        elm = find_element(app, ename, "ElmGenstat") or find_element(app, ename, "ElmSym") or \
              find_element(app, ename, "ElmPvsys") or find_element(app, ename, "ElmLod")
        if not elm:
            continue
        if prev is None:
            # set safe zeros
            _set_element_attr_safe(elm, "pgini", 0.0)
            _set_element_attr_safe(elm, "plinir", 0.0)
            _set_element_attr_safe(elm, "plinis", 0.0)
            _set_element_attr_safe(elm, "plinit", 0.0)
            _set_element_attr_safe(elm, "qgini", 0.0)
            _set_element_attr_safe(elm, "qlinir", 0.0)
            _set_element_attr_safe(elm, "qlinis", 0.0)
            _set_element_attr_safe(elm, "qlinit", 0.0)
            continue
        # restore per-phase if present
        if "plinir" in prev:
            try:
                _set_element_attr_safe(elm, "plinir", prev.get("plinir", 0.0))
                _set_element_attr_safe(elm, "plinis", prev.get("plinis", 0.0))
                _set_element_attr_safe(elm, "plinit", prev.get("plinit", 0.0))
                _set_element_attr_safe(elm, "qlinir", prev.get("qlinir", 0.0))
                _set_element_attr_safe(elm, "qlinis", prev.get("qlinis", 0.0))
                _set_element_attr_safe(elm, "qlinit", prev.get("qlinit", 0.0))
            except Exception:
                pass
        elif "pgini" in prev:
            try:
                _set_element_attr_safe(elm, "pgini", prev.get("pgini", 0.0))
                _set_element_attr_safe(elm, "qgini", prev.get("qgini", 0.0))
            except Exception:
                pass


def apply_solution_with_storage(app, vars_def, x):
    """
    Apply variables (non-storage) and, if storage candidates exist and storage variables defined,
    set storage per-phase values. Returns (candidate, prev_values) or (None, None).
    Respects MIRROR_MAP: for every primary variable its mirror phases are set to the same value.
    Supports symmetric storage mode (storage_P_sym / storage_Q_sym) as well as the legacy
    per-phase mode (storage_P1…P3 / Q1…Q3).
    """
    global STORAGE_CANDIDATES, MIRROR_MAP

    # apply non-storage variables (+ their phase mirrors)
    for idx, var in enumerate(vars_def):
        name = var["name"]
        if name.startswith("storage_"):
            continue
        try:
            val = float(x[idx])
        except Exception:
            continue
        try:
            val = max(var["min"], min(var["max"], val))
        except Exception:
            pass
        set_single_attribute(app, var, val)
        # propagate same value to mirror phases
        if MIRROR_MAP:
            _apply_mirrors(app, var, val, MIRROR_MAP)

    # handle storage vars if defined
    if not STORAGE_CANDIDATES:
        return None, None

    # find storage_node_index
    try:
        idx_node = next(i for i, v in enumerate(vars_def) if v["name"] == "storage_node_index")
    except StopIteration:
        return None, None
    try:
        node_val = int(round(x[idx_node]))
    except Exception:
        node_val = 0
    node_val = max(0, min(len(STORAGE_CANDIDATES) - 1, node_val))
    candidate = STORAGE_CANDIDATES[node_val]

    # --- symmetric storage: one P and one Q for all phases ---
    try:
        idx_psym = next(i for i, v in enumerate(vars_def) if v["name"] == "storage_P_sym")
        P_sym = float(x[idx_psym])
        idx_qsym = next(i for i, v in enumerate(vars_def) if v["name"] == "storage_Q_sym")
        Q_sym = float(x[idx_qsym])
        prev = apply_storage_on_node(app, candidate,
                                     P_sym, P_sym, P_sym,
                                     Q_sym, Q_sym, Q_sym)
        return candidate, prev
    except StopIteration:
        pass

    # --- legacy per-phase storage ---
    try:
        idx_p1 = next(i for i, v in enumerate(vars_def) if v["name"] == "storage_P1")
        P1 = float(x[idx_p1]); P2 = float(x[idx_p1 + 1]); P3 = float(x[idx_p1 + 2])
        Q1 = float(x[idx_p1 + 3]); Q2 = float(x[idx_p1 + 4]); Q3 = float(x[idx_p1 + 5])
    except Exception:
        return candidate, None

    prev = apply_storage_on_node(app, candidate, P1, P2, P3, Q1, Q2, Q3)
    return candidate, prev


# -------------------------
# Load/Set elements from Excel (loads/generators/PV/ES)
# -------------------------
def load_and_set_elements_from_excel(app, file_path):
    """
    Read Excel sheets Loads, Generators, PV, StatGen and set attributes accordingly.
    """
    try:
        loads_df = read_excel_safe(file_path, sheet_name="Loads")
        gens_df = read_excel_safe(file_path, sheet_name="Generators")
        pv_df = read_excel_safe(file_path, sheet_name="PV")
        ES_df = read_excel_safe(file_path, sheet_name="StatGen")
    except Exception as e:
        print(f"Błąd przy wczytywaniu pliku {file_path}: {e}")
        return

    # Loads
    for _, row in loads_df.iterrows():
        name = str(row["name"]).strip()
        try:
            elm = find_element(app, name, "ElmLod")
            if not elm:
                continue
            _set_element_attr_safe(elm, "plinir", float(row["P1"]))
            _set_element_attr_safe(elm, "plinis", float(row["P2"]))
            _set_element_attr_safe(elm, "plinit", float(row["P3"]))
            _set_element_attr_safe(elm, "qlinir", float(row["Q1"]))
            _set_element_attr_safe(elm, "qlinis", float(row["Q2"]))
            _set_element_attr_safe(elm, "qlinit", float(row["Q3"]))
        except Exception as e:
            print(f"Błąd ustawiania Load {name}: {e}")

    # Generators
    for _, row in gens_df.iterrows():
        name = str(row["name"]).strip()
        try:
            elm = find_element(app, name, "ElmSym")
            if not elm:
                continue
            _set_element_attr_safe(elm, "pgini", float(row["P"]))
            _set_element_attr_safe(elm, "qgini", float(row["Q"]))
        except Exception as e:
            print(f"Błąd ustawiania Generator {name}: {e}")

    # PV
    for _, row in pv_df.iterrows():
        name = str(row["name"]).strip()
        try:
            elm = find_element(app, name, "ElmPvsys")
            if not elm:
                continue
            _set_element_attr_safe(elm, "pgini", float(row["P"]))
            _set_element_attr_safe(elm, "qgini", float(row["Q"]))
        except Exception as e:
            print(f"Błąd ustawiania PV {name}: {e}")

    # ES (StatGen)
    for _, row in ES_df.iterrows():
        name = str(row["name"]).strip()
        try:
            elm = find_element(app, name, "ElmGenstat")
            if not elm:
                continue
            _set_element_attr_safe(elm, "pgini", float(row["P"]))
            _set_element_attr_safe(elm, "qgini", float(row["Q"]))
        except Exception as e:
            print(f"Błąd ustawiania ES {name}: {e}")

    print("Parametry Loads, PV, ES i Generators zostały wczytane i ustawione w PowerFactory.")


# -------------------------
# Snapshot function (bus/line/trafo/system)
# -------------------------
def collect_results_snapshot(app):
    results_buses, results_lines, results_trafos, results_sys = [], [], [], []
    try:
        for bus in app.GetCalcRelevantObjects("*.ElmTerm"):
            try:
                results_buses.append({
                    "Bus": bus.loc_name,
                    "U1 [p.u.]": bus.GetAttribute("m:u:A"),
                    "U2 [p.u.]": bus.GetAttribute("m:u:B"),
                    "U3 [p.u.]": bus.GetAttribute("m:u:C"),
                    "U1 [deg]": bus.GetAttribute("m:phiu:A"),
                    "U2 [deg]": bus.GetAttribute("m:phiu:B"),
                    "U3 [deg]": bus.GetAttribute("m:phiu:C"),
                    "UnbFac [%]": bus.GetAttribute("m:ubfac"),
                })
            except Exception:
                continue
    except Exception:
        pass

    try:
        for line in app.GetCalcRelevantObjects("*.ElmLne"):
            try:
                results_lines.append({
                    "Line": line.loc_name,
                    "Loading [%]": line.GetAttribute("c:loading"),
                    "I1 [kA]": line.GetAttribute("m:I:bus1:A"),
                    "I2 [kA]": line.GetAttribute("m:I:bus1:B"),
                    "I3 [kA]": line.GetAttribute("m:I:bus1:C"),
                    "I1 [deg]": line.GetAttribute("m:phii:bus1:A"),
                    "I2 [deg]": line.GetAttribute("m:phii:bus1:B"),
                    "I3 [deg]": line.GetAttribute("m:phii:bus1:C"),
                    "UnbFac [%]": line.GetAttribute("n:ubfac:bus1"),
                    "UnbFacI [%]": line.GetAttribute("m:ubfacI:bus1"),
                    "UnbFacS [%]": line.GetAttribute("m:ubfacS:bus1"),
                })
            except Exception:
                continue
    except Exception:
        pass

    try:
        for tr in app.GetCalcRelevantObjects("*.ElmTr2"):
            try:
                results_trafos.append({
                    "Trafo": tr.loc_name,
                    "Loading [%]": tr.GetAttribute("c:loading"),
                    "I1 [kA]": tr.GetAttribute("m:I:buslv:A"),
                    "I2 [kA]": tr.GetAttribute("m:I:buslv:B"),
                    "I3 [kA]": tr.GetAttribute("m:I:buslv:C"),
                    "I1 [deg]": tr.GetAttribute("m:phii:buslv:A"),
                    "I2 [deg]": tr.GetAttribute("m:phii:buslv:B"),
                    "I3 [deg]": tr.GetAttribute("m:phii:buslv:C"),
                    "UnbFac [%]": tr.GetAttribute("n:ubfac:buslv"),
                    "UnbFacI [%]": tr.GetAttribute("m:ubfacI:buslv"),
                    "UnbFacS [%]": tr.GetAttribute("m:ubfacS:buslv"),
                })
            except Exception:
                continue
    except Exception:
        pass

    try:
        for net in app.GetCalcRelevantObjects("*.ElmNet"):
            try:
                results_sys.append({
                    "System": net.loc_name,
                    "Ploss [MW]": net.GetAttribute("c:LossP"),
                    "Qloss [Mvar]": net.GetAttribute("c:LossQ"),
                    "Pgen [MW]": net.GetAttribute("c:GenP"),
                    "Qgen [Mvar]": net.GetAttribute("c:GenQ"),
                    "Pload [MW]": net.GetAttribute("c:LoadP"),
                    "Qload [Mvar]": net.GetAttribute("c:LoadQ"),
                })
            except Exception:
                continue
    except Exception:
        pass

    return results_buses, results_lines, results_trafos, results_sys


# -------------------------
# Helper: phasor and line current readers
# -------------------------
def _phasor_from_bus(bus):
    """
    Return complex phasors [Va, Vb, Vc] from bus object.
    """
    try:
        ua = bus.GetAttribute("m:u:A"); ub = bus.GetAttribute("m:u:B"); uc = bus.GetAttribute("m:u:C")
        pa = bus.GetAttribute("m:phiu:A"); pb = bus.GetAttribute("m:phiu:B"); pc = bus.GetAttribute("m:phiu:C")
        ph = []
        for mag, ang in ((ua, pa), (ub, pb), (uc, pc)):
            try:
                magf = float(mag); angf = float(ang)
                ph.append(magf * np.exp(1j * np.deg2rad(angf)))
            except Exception:
                ph.append(0+0j)
        return np.array(ph, dtype=complex)
    except Exception:
        return np.array([0+0j, 0+0j, 0+0j], dtype=complex)


def _line_phase_currents(line, end="bus1"):
    """
    Return [Ia,Ib,Ic] (absolute values) for given line terminal (bus1 or bus2).
    """
    names = ["m:I:bus1:A", "m:I:bus1:B", "m:I:bus1:C"] if end == "bus1" else ["m:I:bus2:A", "m:I:bus2:B", "m:I:bus2:C"]
    currents = []
    for n in names:
        try:
            v = line.GetAttribute(n)
            currents.append(abs(float(v)))
        except Exception:
            currents.append(0.0)
    return np.array(currents, dtype=float)


def _get_line_nominal_current(line):
    """
    Try to get nominal current Inom for the line; fallback to compute from loading%.
    """
    try:
        val = line.GetAttribute("c:Inom")
        if val is not None and float(val) > 0:
            return float(val)
    except Exception:
        pass
    try:
        I_ph = _line_phase_currents(line, end="bus1")
        I_actual = np.mean(I_ph[np.where(I_ph > 0)]) if np.any(I_ph > 0) else 0.0
    except Exception:
        I_actual = 0.0
    try:
        loading = line.GetAttribute("c:loading")
        loading = float(loading)
        if loading > 0:
            return max(1e-6, I_actual / (loading / 100.0))
    except Exception:
        pass
    return max(1e-6, I_actual, 1.0)


# -------------------------
# Objective function (5 components + penalty)
# -------------------------
def objective_function(app, vars_def, x, ldf):
    """
    Weighted sum of five components + penalties.
    Robust detection of failed/degenerate loadflows:
    - checks ldf.Execute() return code (if provided)
    - checks fraction of non-zero phase measurements after LF
    If LF appears failed, returns a very large penalty (PENALTY * LARGE_PENALTY_MULTIPLIER).
    """
    global COMPONENT_HISTORY, LARGE_PENALTY_MULTIPLIER

    # Safety cap to avoid float overflow
    LARGE_PENALTY_CAP = 1e300
    # Threshold fraction of non-zero phases required to treat LF as valid
    FRACTION_NONZERO_THRESHOLD = 0.5

    candidate = None
    prev_storage = None
    try:
        candidate, prev_storage = apply_solution_with_storage(app, vars_def, x)

        # Run load flow and catch execution errors
        code = None
        try:
            code = ldf.Execute()
        except Exception as e:
            # failed LDF execution -> very large penalty
            big_pen = min(PENALTY * float(LARGE_PENALTY_MULTIPLIER), LARGE_PENALTY_CAP)
            with open("failed_evals.csv", "a", encoding="utf-8") as f:
                f.write(f"{time.time()},{','.join(map(str, x))},exception_ldf,{str(e)}\n")
            comps = [np.nan] * 5
            COMPONENT_HISTORY.append(comps + [big_pen, big_pen])
            if EVAL_DELAY:
                time.sleep(EVAL_DELAY)
            return big_pen

        # Some PF APIs return a code; treat non-zero as failure (safe check)
        if code is not None:
            try:
                code_num = int(code)
                if code_num != 0:
                    big_pen = min(PENALTY * float(LARGE_PENALTY_MULTIPLIER), LARGE_PENALTY_CAP)
                    with open("failed_evals.csv", "a", encoding="utf-8") as f:
                        f.write(f"{time.time()},{','.join(map(str, x))},ldf_return_code,{code_num}\n")
                    comps = [np.nan] * 5
                    COMPONENT_HISTORY.append(comps + [big_pen, big_pen])
                    if EVAL_DELAY:
                        time.sleep(EVAL_DELAY)
                    return big_pen
            except Exception:
                # if can't interpret code, continue to other checks
                pass

        if EVAL_DELAY:
            time.sleep(EVAL_DELAY)

        # verify nets exist
        nets = app.GetCalcRelevantObjects("*.ElmNet")
        if not nets:
            big_pen = min(PENALTY * float(LARGE_PENALTY_MULTIPLIER), LARGE_PENALTY_CAP)
            comps = [np.nan] * 5
            COMPONENT_HISTORY.append(comps + [big_pen, big_pen])
            return big_pen
        net = nets[0]

        # --- SANITY CHECK: fraction of non-zero phase measurements immediately after LF ---
        try:
            buses_for_check = app.GetCalcRelevantObjects("*.ElmTerm")
            total_nodes = 0
            nonzero_phase_count = 0
            for b in buses_for_check:
                try:
                    ua = b.GetAttribute("m:u:A"); ub = b.GetAttribute("m:u:B"); uc = b.GetAttribute("m:u:C")
                    # convert to floats safely (treat missing as 0)
                    vals = []
                    for v in (ua, ub, uc):
                        try:
                            vf = float(v)
                        except Exception:
                            vf = 0.0
                        vals.append(vf)
                    total_nodes += 1
                    for vf in vals:
                        if abs(vf) > 1e-6:
                            nonzero_phase_count += 1
                except Exception:
                    continue
            if total_nodes == 0:
                big_pen = min(PENALTY * float(LARGE_PENALTY_MULTIPLIER), LARGE_PENALTY_CAP)
                with open("failed_evals.csv", "a", encoding="utf-8") as f:
                    f.write(f"{time.time()},{','.join(map(str, x))},ldf_no_nodes,0\n")
                comps = [np.nan] * 5
                COMPONENT_HISTORY.append(comps + [big_pen, big_pen])
                return big_pen
            fraction_nonzero = nonzero_phase_count / float(3 * total_nodes)
            if fraction_nonzero < FRACTION_NONZERO_THRESHOLD:
                big_pen = min(PENALTY * float(LARGE_PENALTY_MULTIPLIER), LARGE_PENALTY_CAP)
                with open("failed_evals.csv", "a", encoding="utf-8") as f:
                    f.write(f"{time.time()},{','.join(map(str, x))},ldf_insufficient_nonzero_phases,{nonzero_phase_count}/{3*total_nodes}\n")
                comps = [np.nan] * 5
                COMPONENT_HISTORY.append(comps + [big_pen, big_pen])
                return big_pen
        except Exception:
            # if the check fails for unexpected reason, don't block optimization; continue and compute components
            pass

        # --- C1: per-phase voltage deviation
        buses = app.GetCalcRelevantObjects("*.ElmTerm")
        node_count = 0
        sum_sq = 0.0
        for b in buses:
            try:
                ua = b.GetAttribute("m:u:A"); ub = b.GetAttribute("m:u:B"); uc = b.GetAttribute("m:u:C")
                phases = []
                for v in (ua, ub, uc):
                    try:
                        vf = float(v)
                        if vf > 0:
                            phases.append(vf)
                        else:
                            phases.append(None)
                    except Exception:
                        phases.append(None)
                if all(p is None for p in phases):
                    continue
                for pval in phases:
                    if pval is None:
                        continue
                    term = ((pval - 1.05) / 1.05) ** 2
                    sum_sq += term
                node_count += 1
            except Exception:
                continue
        denom = max(1, 3 * node_count)
        C1 = math.sqrt(sum_sq / denom) if denom > 0 else 0.0

        # --- C2: current asymmetry from LINE_NOM_NAME (line "1002")
        line_ref = find_element(app, LINE_NOM_NAME, "ElmLne")
        if line_ref is None:
            C2 = 0.0
        else:
            Iphases = _line_phase_currents(line_ref, end="bus1")
            I_mean = np.mean(Iphases)
            I_unb = math.sqrt(np.mean((Iphases - I_mean) ** 2))
            Inom = _get_line_nominal_current(line_ref)
            C2 = float(I_unb / max(1e-9, Inom))

        # --- C3: active losses normalized by total load
        try:
            Ploss = float(net.GetAttribute("c:LossP"))
        except Exception:
            Ploss = 0.0
        try:
            TotalLoad = float(net.GetAttribute("c:LoadP"))
        except Exception:
            TotalLoad = 0.0
        C3 = float(Ploss / max(1e-9, abs(TotalLoad)))

        # --- C4 and C5: sequence coefficients sums
        a = np.exp(1j * 2.0 * np.pi / 3.0)
        M = (1.0 / 3.0) * np.array([[1, 1, 1], [1, a, a ** 2], [1, a ** 2, a]], dtype=complex)
        sum_v0_coeff = 0.0
        sum_v2_coeff = 0.0
        nodes_considered = 0
        for b in buses:
            ph = _phasor_from_bus(b)
            if np.all(np.abs(ph) == 0):
                continue
            seq = M.dot(ph)
            V0 = seq[0]; V1 = seq[1]; V2 = seq[2]
            V1_mag = max(1e-9, abs(V1))
            sum_v0_coeff += abs(V0) / V1_mag
            sum_v2_coeff += abs(V2) / V1_mag
            nodes_considered += 1
        C4 = float(sum_v0_coeff)
        C5 = float(sum_v2_coeff)

        # --- Penalties for violations (voltage, loading)
        penalty = 0.0
        buses_snap, lines_snap, trafos_snap, _ = collect_results_snapshot(app)
        for b in buses_snap:
            for phase in ["U1 [p.u.]", "U2 [p.u.]", "U3 [p.u.]"]:
                try:
                    if b[phase] < VOLTAGE_MIN or b[phase] > VOLTAGE_MAX:
                        penalty += PENALTY
                except Exception:
                    continue
        for l in lines_snap:
            try:
                if l["Loading [%]"] is not None and float(l["Loading [%]"]) > LOAD_MAX:
                    penalty += PENALTY
            except Exception:
                continue
        for t in trafos_snap:
            try:
                if t["Loading [%]"] is not None and float(t["Loading [%]"]) > LOAD_MAX:
                    penalty += PENALTY
            except Exception:
                continue

        comps = [C1, C2, C3, C4, C5]
        w = WEIGHTS
        if len(w) != 5:
            w = np.array([1.0] * 5) / 5.0
        total_obj = float(np.dot(w, np.array(comps, dtype=float)) + penalty)

        COMPONENT_HISTORY.append([C1, C2, C3, C4, C5, penalty, total_obj])

        return total_obj

    finally:
        # restore storage attributes if they were changed
        if prev_storage is not None and candidate is not None:
            try:
                reset_storage_on_node(app, candidate, prev_storage)
            except Exception:
                with open("storage_apply_errors.log", "a", encoding="utf-8") as f:
                    f.write(f"{time.time()} Błąd przy przywracaniu storage dla {candidate}\n")


# -------------------------
# Surface-scan helper functions
# -------------------------

def get_storage_setpoints_from_x(vars_def, x):
    """
    Extract storage P and Q setpoints from solution vector x.
    Returns (P_list, Q_list):
    - Symmetric mode (storage_P_sym / storage_Q_sym): single-element lists [P_sym], [Q_sym].
    - Legacy per-phase mode (storage_P1..P3 / Q1..Q3): [P1,P2,P3], [Q1,Q2,Q3].
    - No storage variables found: ([], []).
    Pstor_sum* = sum(P_list), Qstor_sum* = sum(Q_list).
    """
    # symmetric first
    try:
        idx_p = next(i for i, v in enumerate(vars_def) if v["name"] == "storage_P_sym")
        idx_q = next(i for i, v in enumerate(vars_def) if v["name"] == "storage_Q_sym")
        return [float(x[idx_p])], [float(x[idx_q])]
    except StopIteration:
        pass
    # per-phase legacy
    try:
        idx_p1 = next(i for i, v in enumerate(vars_def) if v["name"] == "storage_P1")
        P_list = [float(x[idx_p1 + k]) for k in range(3)]
        Q_list = [float(x[idx_p1 + 3 + k]) for k in range(3)]
        return P_list, Q_list
    except (StopIteration, IndexError):
        pass
    return [], []


def set_storage_setpoints_in_x(vars_def, x, P_list, Q_list):
    """
    Return a new copy of x with storage setpoints replaced by P_list and Q_list.
    Symmetric mode: P_list[0] -> storage_P_sym, Q_list[0] -> storage_Q_sym.
    Per-phase mode: P_list[0..2] -> storage_P1..P3, Q_list[0..2] -> storage_Q1..Q3.
    """
    x_new = np.array(x, dtype=float)
    # try symmetric first
    try:
        idx_p = next(i for i, v in enumerate(vars_def) if v["name"] == "storage_P_sym")
        idx_q = next(i for i, v in enumerate(vars_def) if v["name"] == "storage_Q_sym")
        if P_list:
            x_new[idx_p] = float(P_list[0])
        if Q_list:
            x_new[idx_q] = float(Q_list[0])
        return x_new
    except StopIteration:
        pass
    # per-phase legacy
    try:
        idx_p1 = next(i for i, v in enumerate(vars_def) if v["name"] == "storage_P1")
        for k in range(3):
            if k < len(P_list):
                x_new[idx_p1 + k] = float(P_list[k])
            if k < len(Q_list):
                x_new[idx_p1 + 3 + k] = float(Q_list[k])
        return x_new
    except StopIteration:
        pass
    return x_new


def collect_Q_elements(app, exclude_names=None):
    """
    Collect all source elements (ElmSym, ElmPvsys, ElmGenstat) that have a valid
    numeric qgini attribute. Returns list of (name, elm) tuples.
    Elements whose loc_name appears in exclude_names are skipped (used for
    storage elements that are managed separately via x vector).
    """
    if exclude_names is None:
        exclude_names = set()
    result = []
    for pf_class in ("ElmSym", "ElmPvsys", "ElmGenstat"):
        try:
            objs = app.GetCalcRelevantObjects(f"*.{pf_class}")
        except Exception:
            objs = []
        for elm in objs:
            try:
                name = elm.loc_name
                if name in exclude_names:
                    continue
                q = elm.GetAttribute("qgini")
                float(q)  # validates numeric existence
                result.append((name, elm))
            except Exception:
                continue
    return result


def get_qgini_snapshot(elms_list):
    """
    Return dict {name: float(qgini)} for every (name, elm) in elms_list.
    Falls back to 0.0 if attribute reading fails.
    """
    snap = {}
    for name, elm in elms_list:
        try:
            snap[name] = float(elm.GetAttribute("qgini"))
        except Exception:
            snap[name] = 0.0
    return snap


def set_qgini_from_dict(elms_list, q_dict):
    """
    Set qgini on each element whose name appears in q_dict.
    Silently skips elements where the attribute cannot be set.
    """
    for name, elm in elms_list:
        if name in q_dict:
            try:
                _set_element_attr_safe(elm, "qgini", float(q_dict[name]))
            except Exception:
                pass


def compute_weighted_redistribution(values_star, delta, eps=1e-6):
    """
    Distribute a scalar delta among values proportionally to |values_star[i]| + eps.
    If all weights are negligible, distributes evenly.
    Returns numpy array: values_new = values_star + delta * (w_i / sum(w)).
    """
    vs = np.array(values_star, dtype=float)
    weights = np.abs(vs) + eps
    w_sum = float(np.sum(weights))
    if w_sum < 1e-15:
        n = max(1, len(vs))
        weights = np.ones(len(vs)) / n
        w_sum = 1.0
    return vs + delta * (weights / w_sum)


def get_checkpoint_path(method_name):
    """
    Return default checkpoint path for the selected optimizer.
    """
    if CHECKPOINT_FILE:
        return CHECKPOINT_FILE
    method_u = str(method_name).upper()
    fname = f"{method_u.lower()}_checkpoint.npz"
    return os.path.join(os.path.dirname(OUT_FILE), fname)


def load_saved_result(method_name, vars_def):
    """
    Load a previously found optimum without rerunning optimisation.

    Priority:
      1. <method>_checkpoint.npz with keys gbest / gbest_val / best_per_iter
      2. BEST_SOLUTION_FILE aligned by (variable, pf_class, attr)

    Returns dict compatible with optimizer output:
      {"gbest": np.ndarray, "gbest_val": float|nan, "best_per_iter": list}
    or None if nothing usable was found.
    """
    ckpt_path = get_checkpoint_path(method_name)

    if os.path.exists(ckpt_path):
        try:
            with np.load(ckpt_path, allow_pickle=True) as npz:
                gbest = np.array(npz["gbest"], dtype=float)
                if gbest.ndim != 1:
                    gbest = gbest.reshape(-1)
                if len(gbest) != len(vars_def):
                    print(
                        f"Checkpoint ma zły wymiar: {len(gbest)} zamiast {len(vars_def)}. "
                        f"Pomijam {ckpt_path}."
                    )
                else:
                    gbest_val = float(npz["gbest_val"]) if "gbest_val" in npz.files else float("nan")
                    best_per_iter = np.array(npz["best_per_iter"], dtype=float).tolist() if "best_per_iter" in npz.files else []
                    print(f"Wczytano optimum z checkpointu: {ckpt_path}")
                    return {
                        "gbest": gbest,
                        "gbest_val": gbest_val,
                        "best_per_iter": best_per_iter,
                    }
        except Exception as e:
            print(f"Nie udało się wczytać checkpointu {ckpt_path}: {e}")

    if os.path.exists(BEST_SOLUTION_FILE):
        try:
            loaded = {}
            with open(BEST_SOLUTION_FILE, "r", encoding="utf-8") as f:
                for raw in f:
                    line = raw.strip()
                    normalized = " ".join(line.split())
                    if not line or normalized.startswith("Best solution") or normalized.startswith("Storage elements"):
                        continue
                    parts = line.split()
                    if len(parts) < 3:
                        continue
                    name = parts[0]
                    try:
                        val = float(parts[1])
                    except Exception:
                        continue
                    pf_class = parts[2]
                    attr = parts[3] if len(parts) >= 4 else ""
                    loaded[(name, pf_class, attr)] = val

            x_vals = []
            missing = []
            for var in vars_def:
                key = (var.get("name", ""), var.get("pf_class", ""), var.get("attr", ""))
                if key in loaded:
                    x_vals.append(float(loaded[key]))
                else:
                    missing.append(key)

            if not missing:
                print(f"Wczytano optimum z pliku tekstowego: {BEST_SOLUTION_FILE}")
                return {
                    "gbest": np.array(x_vals, dtype=float),
                    "gbest_val": float("nan"),
                    "best_per_iter": [],
                }

            print(
                f"Plik {BEST_SOLUTION_FILE} nie pasuje do bieżącej konfiguracji zmiennych "
                f"(brakuje {len(missing)} pozycji)."
            )
        except Exception as e:
            print(f"Nie udało się wczytać {BEST_SOLUTION_FILE}: {e}")

    return None


# -------------------------
# 2D surface scan + plots
# -------------------------

def _plot_surface_3d(dP_arr, dQ_arr, J_surface, J_star, out_dir):
    """
    Save a 3D surface PNG of J(ΔP, ΔQ) with the optimum point marked.
    """
    try:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        PP, QQ = np.meshgrid(dP_arr, dQ_arr)
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(PP, QQ, J_surface, cmap="viridis", alpha=0.85, edgecolor="none")
        ax.scatter([0.0], [0.0], [J_star], color="red", s=100, zorder=6, label="Optimum (ΔP=0, ΔQ=0)")
        ax.set_xlabel("ΔP_stor_sum [kW]")
        ax.set_ylabel("ΔQ_total_sum [kvar]")
        ax.set_zlabel("J (funkcja celu)")
        ax.set_title("Powierzchnia funkcji celu 3D")
        fig.colorbar(surf, ax=ax, shrink=0.45, label="J")
        ax.legend()
        plt.tight_layout()
        fpath = os.path.join(out_dir, "Surface_3D.png")
        plt.savefig(fpath, dpi=200)
        plt.show()
        plt.close(fig)
        print(f"Wykres 3D zapisany: {fpath}")
    except Exception as e:
        print(f"Błąd wykresu 3D: {e}")
        traceback.print_exc()


def _plot_surface_heatmap(dP_arr, dQ_arr, J_surface, J_star, out_dir):
    """
    Save a 2D contourf heatmap PNG of J(ΔP, ΔQ).
    """
    try:
        PP, QQ = np.meshgrid(dP_arr, dQ_arr)
        fig, ax = plt.subplots(figsize=(8, 6))
        cf = ax.contourf(PP, QQ, J_surface, levels=30, cmap="viridis")
        ax.contour(PP, QQ, J_surface, levels=30, colors="k", linewidths=0.3, alpha=0.4)
        plt.colorbar(cf, ax=ax, label="J")
        ax.scatter([0.0], [0.0], color="red", s=100, zorder=6, label="Optimum (ΔP=0, ΔQ=0)")
        ax.set_xlabel("ΔP_stor_sum [kW]")
        ax.set_ylabel("ΔQ_total_sum [kvar]")
        ax.set_title("Mapa ciepła funkcji celu")
        ax.legend()
        plt.tight_layout()
        fpath = os.path.join(out_dir, "Surface_Heatmap.png")
        plt.savefig(fpath, dpi=200)
        plt.show()
        plt.close(fig)
        print(f"Mapa ciepła zapisana: {fpath}")
    except Exception as e:
        print(f"Błąd mapy ciepła: {e}")
        traceback.print_exc()


def run_surface_scan(app, vars_def, ldf, res):
    """
    Perform a 2D grid scan of the objective function in the space
    (ΔP_stor_sum, ΔQ_total_sum) around the optimum x*.

    Axes:
      X – ΔP_stor_sum : total delta of storage active-power setpoints
      Y – ΔQ_total_sum: total delta of reactive power of sources + storage

    Algorithm per grid point (dP, dQ):
      1. Redistribute dP among storage P variables proportionally to |P*| + eps.
      2. Redistribute dQ among ALL Q variables (ElmSym/ElmPvsys/ElmGenstat + storage Q)
         proportionally to |Q*| + eps.
      3. Clip storage values to STORAGE_P/Q_MIN/MAX.
      4. Build x_scan (x* with new storage P/Q), set qgini on source elements.
      5. Call objective_function(app, vars_def, x_scan, ldf) – internally applies +
         restores storage; Q sources are restored manually after the call.

    Outputs (written to same folder as OUT_FILE):
      - OUT_SURFACE_FILE  (.xlsx): sheets Axis_dP, Axis_dQ, Surface_J, Surface_meta
      - surface_scan.npz  : NumPy archive {dP, dQ, J, Pstor_sum_star, Qtot_sum_star, J_star}
      - Surface_3D.png    : 3-D surface plot
      - Surface_Heatmap.png: 2-D contourf heatmap

    Returns (dP_arr, dQ_arr, J_surface) or None if skipped/failed.
    """
    global STORAGE_CANDIDATES

    if not SURFACE_ENABLED:
        print("SURFACE_ENABLED=False – pomijam skanowanie 2D.")
        return None

    print("=" * 60)
    print("=== Tryb skanowania 2D powierzchni funkcji celu ===")
    print(f"    Siatka: {GRID_NQ} x {GRID_NP}  ({GRID_NQ * GRID_NP} punktów)")
    print(f"    ΔP ∈ [{DP_MIN}, {DP_MAX}]  ΔQ ∈ [{DQ_MIN}, {DQ_MAX}]")
    print("=" * 60)

    if res is None or res.get("gbest") is None:
        print("Brak wyznaczonego optimum (res/gbest). Pomijam skanowanie.")
        return None

    x_star = np.array(res["gbest"], dtype=float)

    # --- Apply x* to PowerFactory and run LF to set baseline state ---
    try:
        apply_solution_with_storage(app, vars_def, x_star)
        ldf.Execute()
    except Exception as e:
        print(f"Błąd ustawiania stanu optimum przed skanem: {e}")
        return None

    # --- Extract storage P*/Q* from x* ---
    P_star_list, Q_star_list = get_storage_setpoints_from_x(vars_def, x_star)
    has_storage = len(P_star_list) > 0

    Pstor_sum_star = sum(P_star_list) if has_storage else 0.0
    Qstor_sum_star = sum(Q_star_list) if has_storage else 0.0

    if not has_storage:
        print("Brak zmiennych magazynu (storage) w vars_def. Skan będzie tylko po ΔQ źródeł.")

    # --- Determine names of storage elements to exclude from Q-source collection ---
    storage_elem_names = set()
    if has_storage and STORAGE_CANDIDATES:
        try:
            idx_node = next(i for i, v in enumerate(vars_def) if v["name"] == "storage_node_index")
            node_val = max(0, min(len(STORAGE_CANDIDATES) - 1, int(round(float(x_star[idx_node])))))
            cand = STORAGE_CANDIDATES[node_val]
            for k in ("elem_A", "elem_B", "elem_C"):
                n = cand.get(k, "")
                if n:
                    storage_elem_names.add(n)
        except (StopIteration, Exception) as e:
            print(f"  Uwaga: nie udało się odczytać kandydata storage: {e}")

    # --- Collect Q source elements and snapshot their Q* ---
    q_elms = collect_Q_elements(app, exclude_names=storage_elem_names)
    if not q_elms:
        print("Brak elementów Q (ElmSym/ElmPvsys/ElmGenstat) w modelu. Pomijam skanowanie.")
        return None

    q_star_dict = get_qgini_snapshot(q_elms)
    Q_src_star = np.array([q_star_dict[name] for name, _ in q_elms], dtype=float)
    Q_src_sum_star = float(np.sum(Q_src_star))

    # Combined Q vector for redistribution: [Q_sources..., Q_storage_representatives...]
    # Symmetric storage: one representative Q_sym; per-phase: three Q values.
    if has_storage:
        Q_stor_repr_star = np.array(Q_star_list, dtype=float)   # [Q_sym] or [Q1,Q2,Q3]
    else:
        Q_stor_repr_star = np.array([], dtype=float)

    Q_combined_star = np.concatenate([Q_src_star, Q_stor_repr_star])
    n_src = len(Q_src_star)

    Qtot_sum_star = Q_src_sum_star + Qstor_sum_star
    J_star = float(res.get("gbest_val", np.nan))

    print(f"Pstor_sum*  = {Pstor_sum_star:.6f}  kW")
    print(f"Qtot_sum*   = {Qtot_sum_star:.8f}  kvar  "
          f"(źródła: {Q_src_sum_star:.6f}, magazyn: {Qstor_sum_star:.6f})")
    print(f"J*          = {J_star:.6f}")
    print(f"Elementów Q (źródła): {n_src}, storage repr.: {len(Q_stor_repr_star)}")

    # --- Build scan grid ---
    dP_arr = np.linspace(DP_MIN, DP_MAX, GRID_NP)
    dQ_arr = np.linspace(DQ_MIN, DQ_MAX, GRID_NQ)
    J_surface = np.full((GRID_NQ, GRID_NP), np.nan)

    total_pts = GRID_NQ * GRID_NP
    done = 0
    t0 = time.time()
    print_every = max(1, total_pts // 20)

    # === Main scan loop ===
    for i_q, dQ in enumerate(dQ_arr):
        for j_p, dP in enumerate(dP_arr):

            # ---- Redistribute P among storage elements ----
            if has_storage:
                P_new_repr = compute_weighted_redistribution(P_star_list, dP, EPS_WEIGHT)
                P_new_repr = np.clip(P_new_repr, STORAGE_P_MIN, STORAGE_P_MAX)
                P_new_list = P_new_repr.tolist()
            else:
                P_new_list = []

            # ---- Redistribute Q among sources + storage ----
            Q_new_combined = compute_weighted_redistribution(Q_combined_star, dQ, EPS_WEIGHT)
            Q_new_src   = Q_new_combined[:n_src]
            Q_new_stor_repr = Q_new_combined[n_src:]

            if has_storage:
                Q_new_stor_repr = np.clip(Q_new_stor_repr, STORAGE_Q_MIN, STORAGE_Q_MAX)
                Q_new_list = Q_new_stor_repr.tolist()
            else:
                Q_new_list = []

            # ---- Build x_scan (x* with replaced storage P/Q) ----
            if has_storage:
                x_scan = set_storage_setpoints_in_x(vars_def, x_star, P_new_list, Q_new_list)
            else:
                x_scan = np.array(x_star, dtype=float)

            # ---- Apply new Q to source elements in PF model ----
            q_new_dict = {name: float(Q_new_src[k]) for k, (name, _) in enumerate(q_elms)}
            set_qgini_from_dict(q_elms, q_new_dict)

            # ---- Evaluate objective (internally applies storage from x_scan, runs LF, restores storage) ----
            try:
                J_val = objective_function(app, vars_def, x_scan, ldf)
            except Exception as e:
                J_val = np.nan
                print(f"  [scan i={i_q},j={j_p}] Wyjątek: {e}")

            J_surface[i_q, j_p] = J_val

            # ---- Restore Q sources back to baseline ----
            set_qgini_from_dict(q_elms, q_star_dict)

            done += 1
            if done % print_every == 0 or done == total_pts:
                elapsed = time.time() - t0
                eta = (elapsed / done) * (total_pts - done) if done > 0 else 0.0
                print(f"  Skan: {done}/{total_pts} ({100.0 * done / total_pts:.0f}%), "
                      f"czas: {elapsed:.1f}s, ETA: {eta:.1f}s, "
                      f"J_min_dotąd={np.nanmin(J_surface):.5g}")

    # === Restore model to optimum state after the loop ===
    try:
        apply_solution_with_storage(app, vars_def, x_star)
        set_qgini_from_dict(q_elms, q_star_dict)
        ldf.Execute()
        print("Model przywrócony do stanu optimum po skanowaniu.")
    except Exception as e:
        print(f"Uwaga: błąd przywracania stanu optimum po skanowaniu: {e}")

    # --- Stats ---
    J_min_scan = float(np.nanmin(J_surface)) if not np.all(np.isnan(J_surface)) else np.nan
    j_star_idx = int(np.argmin(np.abs(dP_arr)))
    i_star_idx = int(np.argmin(np.abs(dQ_arr)))
    J_center = float(J_surface[i_star_idx, j_star_idx])
    print(f"Skanowanie zakończone. J_min w siatce = {J_min_scan:.6f}, J w (0,0) = {J_center:.6f}")

    out_dir = os.path.dirname(OUT_FILE)

    # === Save Excel ===
    try:
        df_dP   = pd.DataFrame({"j": range(GRID_NP), "dP": dP_arr})
        df_dQ   = pd.DataFrame({"i": range(GRID_NQ), "dQ": dQ_arr})
        df_J    = pd.DataFrame(
            J_surface,
            index=[f"dQ={v:.4g}" for v in dQ_arr],
            columns=[f"dP={v:.4g}" for v in dP_arr],
        )
        meta_keys = [
            "date", "METHOD", "OBJECTIVE",
            "GRID_NP", "GRID_NQ",
            "DP_MIN", "DP_MAX", "DQ_MIN", "DQ_MAX", "EPS_WEIGHT",
            "Pstor_sum_star", "Qtot_sum_star", "J_star", "J_center", "J_min_scan",
        ]
        meta_vals = [
            str(pd.Timestamp.now()), METHOD, OBJECTIVE,
            GRID_NP, GRID_NQ,
            DP_MIN, DP_MAX, DQ_MIN, DQ_MAX, EPS_WEIGHT,
            Pstor_sum_star, Qtot_sum_star, J_star, J_center, J_min_scan,
        ]
        df_meta = pd.DataFrame({"key": meta_keys, "value": meta_vals})

        with pd.ExcelWriter(OUT_SURFACE_FILE, engine="openpyxl") as writer:
            df_dP.to_excel(writer,  sheet_name="Axis_dP",      index=False)
            df_dQ.to_excel(writer,  sheet_name="Axis_dQ",      index=False)
            df_J.to_excel(writer,   sheet_name="Surface_J",    index=True)
            df_meta.to_excel(writer, sheet_name="Surface_meta", index=False)
        print(f"Wyniki powierzchni (Excel) zapisane: {OUT_SURFACE_FILE}")

        # NPZ archive
        npz_path = os.path.splitext(OUT_SURFACE_FILE)[0] + ".npz"
        np.savez(
            npz_path,
            dP=dP_arr, dQ=dQ_arr, J=J_surface,
            Pstor_sum_star=np.array([Pstor_sum_star]),
            Qtot_sum_star=np.array([Qtot_sum_star]),
            J_star=np.array([J_star]),
        )
        print(f"NPZ zapisany: {npz_path}")
    except Exception as e:
        print(f"Błąd zapisu wyników powierzchni: {e}")
        traceback.print_exc()

    # === Plots ===
    _plot_surface_3d(dP_arr, dQ_arr, J_surface, J_star, out_dir)
    _plot_surface_heatmap(dP_arr, dQ_arr, J_surface, J_star, out_dir)

    return dP_arr, dQ_arr, J_surface


# -------------------------
# MAIN
# -------------------------
def main():
    global STORAGE_CANDIDATES, COMPONENT_HISTORY, MIRROR_MAP
    # ensure PowerFactory available
    if powerfactory is None:
        print("powerfactory package not available. This script must run in PowerFactory Python environment.")
        return

    app = powerfactory.GetApplicationExt(USER)
    app.ActivateProject(PROJECT_NAME)
    ldf = app.GetFromStudyCase("ComLdf")

    excel_available = True
    vars_from_best_solution = False

    if RUN_OPTIMIZATION:
        vars_def = load_vars(EXCEL_FILE)
    else:
        try:
            vars_def = load_vars_from_best_solution(BEST_SOLUTION_FILE)
            vars_from_best_solution = True
            excel_available = False
            print(f"Tryb tylko płaszczyzna: używam {BEST_SOLUTION_FILE} zamiast Excela.")
        except Exception as e_best:
            print(f"Uwaga: nie udało się odczytać {BEST_SOLUTION_FILE}: {e_best}")
            print(f"Próbuję fallback do Excela: {EXCEL_FILE}")
            vars_def = load_vars(EXCEL_FILE)
            excel_available = True

    print(f"Wczytano {len(vars_def)} zmiennych do optymalizacji.")

    # --- Symmetric phase grouping ---
    if SYMMETRIC_PHASES and not vars_from_best_solution:
        vars_def, MIRROR_MAP = build_symmetric_vars(vars_def)
        n_mirrors = sum(len(v) for v in MIRROR_MAP.values())
        print(f"Tryb symetryczny: zredukowano do {len(vars_def)} zmiennych "
              f"(usunięto {n_mirrors} duplikatów faz). "
              f"Grup: {len(MIRROR_MAP)}.")
    else:
        MIRROR_MAP = {}
        if vars_from_best_solution:
            print("Tryb awaryjny: pomijam redukcję symetryczną, zachowuję zmienne z best_solution.txt.")

    # Load storage candidates and append storage variables to vars_def if present
    if excel_available:
        STORAGE_CANDIDATES = load_storage_candidates(EXCEL_FILE)
    else:
        STORAGE_CANDIDATES = load_storage_candidates_from_best_solution(BEST_SOLUTION_FILE)

    K = len(STORAGE_CANDIDATES)
    has_storage_vars_already = any(v.get("name", "").startswith("storage_") for v in vars_def)
    if K > 0 and not has_storage_vars_already:
        vars_def.append({
            "name": "storage_node_index",
            "pf_class": "choice",
            "attr": "",
            "min": 0.0,
            "max": float(max(0, K - 1))
        })
        if SYMMETRIC_STORAGE:
            # one P and one Q for all 3 storage phases
            vars_def += [
                {"name": "storage_P_sym", "pf_class": "storage", "attr": "", "min": STORAGE_P_MIN, "max": STORAGE_P_MAX},
                {"name": "storage_Q_sym", "pf_class": "storage", "attr": "", "min": STORAGE_Q_MIN, "max": STORAGE_Q_MAX},
            ]
            print(f"Dodano SYMETRYCZNE zmienne magazynu (kandydatów: {K}, P_sym + Q_sym). ")
        else:
            vars_def += [
                {"name": "storage_P1", "pf_class": "storage", "attr": "", "min": STORAGE_P_MIN, "max": STORAGE_P_MAX},
                {"name": "storage_P2", "pf_class": "storage", "attr": "", "min": STORAGE_P_MIN, "max": STORAGE_P_MAX},
                {"name": "storage_P3", "pf_class": "storage", "attr": "", "min": STORAGE_P_MIN, "max": STORAGE_P_MAX},
                {"name": "storage_Q1", "pf_class": "storage", "attr": "", "min": STORAGE_Q_MIN, "max": STORAGE_Q_MAX},
                {"name": "storage_Q2", "pf_class": "storage", "attr": "", "min": STORAGE_Q_MIN, "max": STORAGE_Q_MAX},
                {"name": "storage_Q3", "pf_class": "storage", "attr": "", "min": STORAGE_Q_MIN, "max": STORAGE_Q_MAX},
            ]
            print(f"Dodano zmienne magazynu per-faza (kandydatów: {K}).")
    elif K > 0 and has_storage_vars_already:
        print(f"Używam zmiennych storage odtworzonych z best_solution.txt (kandydatów: {K}).")

    # Set initial elements from Excel sheets
    if excel_available:
        load_and_set_elements_from_excel(app, EXCEL_FILE)
    else:
        print("Uwaga: pomijam wczytanie Loads/Generators/PV/StatGen z Excela (Excel niedostępny).")

    # initial snapshot and save to START_FILE
    try:
        ldf.Execute()
    except Exception:
        pass
    buses_before, lines, trafos, sys_before = collect_results_snapshot(app)
    df_buses_before = pd.DataFrame(buses_before)
    df_lines = pd.DataFrame(lines)
    df_trafos = pd.DataFrame(trafos)
    df_sys = pd.DataFrame(sys_before)
    with pd.ExcelWriter(START_FILE, engine="openpyxl") as writer:
        df_buses_before.to_excel(writer, sheet_name="Buses_Before", index=False)
        df_lines.to_excel(writer, sheet_name="Lines", index=False)
        df_trafos.to_excel(writer, sheet_name="Transformers", index=False)
        df_sys.to_excel(writer, sheet_name="System_Before", index=False)
    print(f"Wyniki początkowe zapisane do {START_FILE}")

    # Prepare objective closure
    Dim = len(vars_def)
    Lb = np.array([v["min"] for v in vars_def])
    Ub = np.array([v["max"] for v in vars_def])

    # ---- Progress-tracking wrapper ----
    # Variables held in lists so the inner closure can mutate them.
    _eval_counter = [0]
    _best_val     = [np.inf]
    _t_start_obj  = [None]   # filled just before optimizer starts
    TOTAL_EVALS   = N_ITER * N_PARTICLES  # expected (approximate for CEO)

    def obj(x):
        val = objective_function(app, vars_def, x, ldf)
        _eval_counter[0] += 1
        n = _eval_counter[0]
        t0 = _t_start_obj[0] or time.time()

        # ✅ New overall minimum – print full component breakdown
        if val < _best_val[0]:
            _best_val[0] = val
            comps = COMPONENT_HISTORY[-1] if COMPONENT_HISTORY else [np.nan] * 7
            c1, c2, c3, c4, c5, pen, _ = comps
            print(
                f"  ✅ NOWE MINIMUM #{n:>6d}: f={val:.6f}  "
                f"C1={c1:.4f}  C2={c2:.4f}  C3={c3:.4f}  "
                f"C4={c4:.4f}  C5={c5:.4f}  kara={pen:.0f}"
            )
            try:
                app.PrintPlain(f"BEST #{n}: f={val:.6f}")
            except Exception:
                pass

        # 📊 Periodic status line
        if n % PRINT_EVERY == 0:
            elapsed = time.time() - t0
            rate    = n / elapsed if elapsed > 0 else 0.0
            eta_s   = (TOTAL_EVALS - n) / rate if rate > 0 else 0.0
            pct     = 100.0 * n / TOTAL_EVALS if TOTAL_EVALS > 0 else 0.0
            print(
                f"  [{n:>6d}/{TOTAL_EVALS}  {pct:5.1f}%]  "
                f"f_best={_best_val[0]:.6f}  "
                f"czas={elapsed:.0f}s  ETA≈{eta_s:.0f}s  "
                f"({rate:.2f} eval/s)"
            )

        return val

    time_start = time.time()
    # Choose optimizer
    res = None
    METHOD_U = METHOD.upper()

    # ---- Pre-run information ----
    TOTAL_EVALS = N_ITER * N_PARTICLES  # recompute in outer scope for clarity
    print(f"\n{'='*65}")
    print(f"  Algorytm  : {METHOD_U}")
    print(f"  Wymiar    : {Dim}  |  Cząstki: {N_PARTICLES}  |  Iteracje: {N_ITER}")
    print(f"  Tryb      : {'optymalizacja + płaszczyzna' if RUN_OPTIMIZATION else 'tylko płaszczyzna'}")
    if RUN_OPTIMIZATION:
        print(f"  Oczekiwane ewaluacje: ~{TOTAL_EVALS:,} ({N_ITER} × {N_PARTICLES})")
        print(f"  Raport co: {PRINT_EVERY} ewaluacji")
    else:
        print(f"  Źródło x* : {get_checkpoint_path(METHOD_U)} lub {BEST_SOLUTION_FILE}")
    print(f"  Cel       : {OBJECTIVE}")
    if SURFACE_ENABLED:
        print(f"  Skan 2D   : {GRID_NQ}×{GRID_NP} = {GRID_NQ*GRID_NP} punktów")
    print(f"{'='*65}\n")
    _t_start_obj[0] = time_start

    if RUN_OPTIMIZATION:
        print(f"Uruchamiam metodę: {METHOD_U}")
        if METHOD_U == "PSO":
            pso = PSO(obj, N_PARTICLES, Dim, Lb, Ub, N_ITER, W, C1, C2,
                      autosave_every_iters=5, autosave_path="pso_checkpoint.npz", eval_delay=EVAL_DELAY)
            res = pso.optimize()
        elif METHOD_U == "CEO":
            max_fes = N_ITER * N * N_PARTICLES
            ceo = CEO(obj, N_PARTICLES, Dim, Lb, Ub, N, max_fes)
            res = ceo.optimize()
        elif METHOD_U == "PO":
            po = PO(obj, N_PARTICLES, Dim, Lb, Ub, N_ITER,
                    autosave_every_iters=5, autosave_path="po_checkpoint.npz", eval_delay=EVAL_DELAY)
            res = po.optimize()
        elif METHOD_U == "DOA":
            doa = DOA(obj, N_PARTICLES, Dim, Lb, Ub, N_ITER,
                      autosave_every_iters=5, autosave_path="doa_checkpoint.npz", eval_delay=EVAL_DELAY)
            res = doa.optimize()
        elif METHOD_U == "RTO":
            rto = RTO(obj, N_PARTICLES, Dim, Lb, Ub, N_ITER,
                      autosave_every_iters=5, autosave_path="rto_checkpoint.npz", eval_delay=EVAL_DELAY)
            res = rto.optimize()
        elif METHOD_U == "SPO":
            spo = SPO(obj, N_PARTICLES, Dim, Lb, Ub, N_ITER,
                      autosave_every_iters=5, autosave_path="spo_checkpoint.npz", eval_delay=EVAL_DELAY)
            res = spo.optimize()
        elif METHOD_U == "BPB":
            bpb = BPB(obj, N_PARTICLES, Dim, Lb, Ub, N_ITER,
                      autosave_every_iters=5, autosave_path="bpb_checkpoint.npz", eval_delay=EVAL_DELAY)
            res = bpb.optimize()
        elif METHOD_U == "KLA":
            kla = KLA(obj, N_PARTICLES, Dim, Lb, Ub, N_ITER + 10,
                      autosave_every_iters=5, autosave_path="kla_checkpoint.npz", eval_delay=EVAL_DELAY)
            res = kla.optimize()
        elif METHOD_U == "KEO":
            keo = KEO(obj, N_PARTICLES, Dim, Lb, Ub, N_ITER,
                      autosave_every_iters=5, autosave_path="keo_checkpoint.npz", eval_delay=EVAL_DELAY)
            res = keo.optimize()
        elif METHOD_U == "DOE":
            doe = DOE(obj, N_PARTICLES, Dim, Lb, Ub, N_ITER,
                      autosave_every_iters=5, autosave_path="doe_checkpoint.npz", eval_delay=EVAL_DELAY)
            res = doe.optimize()
        elif METHOD_U == "OSA":
            osa = OSA(obj, N_PARTICLES, Dim, Lb, Ub, N_ITER,
                      autosave_every_iters=5, autosave_path="osa_checkpoint.npz", eval_delay=EVAL_DELAY)
            res = osa.optimize()
        elif METHOD_U == "ECO":
            eco = ECO(obj, N_PARTICLES, Dim, Lb, Ub, N_ITER,
                      autosave_every_iters=5, autosave_path="eco_checkpoint.npz", eval_delay=EVAL_DELAY)
            res = eco.optimize()
        elif METHOD_U == "GOA":
            goa = GOA(obj, N_PARTICLES, Dim, Lb, Ub, N_ITER,
                      autosave_every_iters=5, autosave_path="goa_checkpoint.npz", eval_delay=EVAL_DELAY)
            res = goa.optimize()
        elif METHOD_U == "COO":
            coo = COO(obj, N_PARTICLES, Dim, Lb, Ub, N_ITER,
                      autosave_every_iters=5, autosave_path="coo_checkpoint.npz", eval_delay=EVAL_DELAY)
            res = coo.optimize()
        else:
            # fallback to a simple differential evolution (for offline tests)
            print("Unknown METHOD - falling back to differential_evolution (scipy)")
            def wrapped(x):
                return obj(np.array(x, dtype=float))
            bounds = [(float(Lb[i]), float(Ub[i])) for i in range(Dim)]
            out = differential_evolution(wrapped, bounds, maxiter=max(1, N_ITER))
            res = {"gbest": np.array(out.x), "gbest_val": float(out.fun), "best_per_iter": [float(out.fun)]}
    else:
        print("Pomijam optymalizację. Wczytuję zapisane optimum dla skanu płaszczyzny...")
        res = load_saved_result(METHOD_U, vars_def)

    time_end = time.time()
    elapsed_total = time_end - time_start
    n_done = _eval_counter[0]

    print(f"\n{'='*65}")
    print(f"  ✅ {'OPTYMALIZACJA ZAKOŃCZONA' if RUN_OPTIMIZATION else 'WCZYTANO ZAPISANE OPTIMUM'}")
    print(f"     Metoda        : {METHOD_U}")
    print(f"     Czas          : {elapsed_total:.1f} s  ({elapsed_total/60:.1f} min)")
    if RUN_OPTIMIZATION:
        print(f"     Ewaluacji     : {n_done:,}  (oczekiwano ~{TOTAL_EVALS:,})")
    else:
        print(f"     Ewaluacji     : {n_done:,}  (bez ponownej optymalizacji)")
    if RUN_OPTIMIZATION and n_done > 0:
        print(f"     Śr. czas/eval : {elapsed_total/n_done:.3f} s")
    if res is not None:
        print(f"     f_best        : {res.get('gbest_val', float('nan')):.6f}")
    print(f"{'='*65}\n")

    try:
        app.PrintError("running time:{:.5f}".format(elapsed_total))
    except Exception:
        pass

    # safety: ensure res exists
    if res is None:
        print("Brak zapisanego optimum. Najpierw trzeba mieć poprawny checkpoint lub zgodny plik best_solution.txt.")
        return

    # save objective history text (fast preview)
    lista = res.get("best_per_iter", [])
    with open("lista.txt", "w", encoding="utf-8") as f:
        for el in lista:
            f.write(f"{el}\n")

    # Apply final solution (with storage) before taking final snapshots
    try:
        candidate, prev = apply_solution_with_storage(app, vars_def, res.get("gbest"))
    except Exception:
        candidate = None
        prev = None
        try:
            apply_solution(app, vars_def, res.get("gbest"))
        except Exception:
            pass

    # Execute final loadflow and collect final snapshots
    try:
        ldf.Execute()
    except Exception:
        pass
    buses_after, lines, trafos, sys_after = collect_results_snapshot(app)

    # Prepare df_vars (final variable values)
    df_vars = pd.DataFrame(columns=["variable", "value", "pf_class", "attr"])
    try:
        if res.get("gbest") is not None:
            rows = []
            gbest = res["gbest"]
            for idx, var in enumerate(vars_def):
                name = var.get("name", "")
                try:
                    val = float(gbest[idx])
                except Exception:
                    val = gbest[idx] if idx < len(gbest) else None
                rows.append({
                    "variable": name,
                    "value": val,
                    "pf_class": var.get("pf_class", ""),
                    "attr": var.get("attr", "")
                })
            df_vars = pd.DataFrame(rows)
    except Exception:
        df_vars = pd.DataFrame(columns=["variable", "value", "pf_class", "attr"])

    # Storage elements attributes after applying best solution
    df_storage_elems = pd.DataFrame(columns=["elem_name", "phase", "pgini", "qgini", "plinir", "plinis", "plinit", "qlinir", "qlinis", "qlinit"])
    try:
        if candidate is not None:
            elems = [("A", candidate.get("elem_A")), ("B", candidate.get("elem_B")), ("C", candidate.get("elem_C"))]
            rows_e = []
            for phase, ename in elems:
                if not ename:
                    continue
                elm = find_element(app, ename, "ElmGenstat") or find_element(app, ename, "ElmSym") or \
                      find_element(app, ename, "ElmPvsys") or find_element(app, ename, "ElmLod")
                if not elm:
                    continue
                def safe_get(attr):
                    try:
                        return elm.GetAttribute(attr)
                    except Exception:
                        return None
                row_e = {
                    "elem_name": ename,
                    "phase": phase,
                    "pgini": safe_get("pgini"),
                    "qgini": safe_get("qgini"),
                    "plinir": safe_get("plinir"),
                    "plinis": safe_get("plinis"),
                    "plinit": safe_get("plinit"),
                    "qlinir": safe_get("qlinir"),
                    "qlinis": safe_get("qlinis"),
                    "qlinit": safe_get("qlinit"),
                }
                rows_e.append(row_e)
            if rows_e:
                df_storage_elems = pd.DataFrame(rows_e)
    except Exception:
        df_storage_elems = pd.DataFrame(columns=df_storage_elems.columns)

    # Restore storage to previous values (if any)
    try:
        if candidate is not None:
            reset_storage_on_node(app, candidate, prev if prev is not None else {})
    except Exception:
        pass

    # Prepare DataFrames for final Excel output
    df_buses_after = pd.DataFrame(buses_after)
    df_lines = pd.DataFrame(lines)
    df_trafos = pd.DataFrame(trafos)
    df_sys = pd.DataFrame(sys_after)
    df_eval = pd.DataFrame(res.get("best_per_iter", []), columns=["Best_per_iter"]) if res.get("best_per_iter") else pd.DataFrame()

    # COMPONENT_HISTORY -> DataFrame
    try:
        if COMPONENT_HISTORY:
            cols = ["C1_Udev_ph", "C2_Iunb_norm", "C3_Ploss_norm", "C4_V0_sum", "C5_V2_sum", "penalty", "total_obj"]
            df_comp_hist = pd.DataFrame(COMPONENT_HISTORY, columns=cols)
        else:
            df_comp_hist = pd.DataFrame(columns=["C1_Udev_ph", "C2_Iunb_norm", "C3_Ploss_norm", "C4_V0_sum", "C5_V2_sum", "penalty", "total_obj"])
    except Exception:
        df_comp_hist = pd.DataFrame(columns=["C1_Udev_ph", "C2_Iunb_norm", "C3_Ploss_norm", "C4_V0_sum", "C5_V2_sum", "penalty", "total_obj"])

    # Best components (recompute for final gbest to be sure)
    best_components = None
    try:
        if res.get("gbest") is not None:
            _ = objective_function(app, vars_def, res["gbest"], ldf)
            if COMPONENT_HISTORY:
                last = COMPONENT_HISTORY[-1]
                best_components = dict(zip(["C1", "C2", "C3", "C4", "C5", "penalty", "total"], last))
    except Exception:
        best_components = None

    # Write Excel with multiple sheets
    with pd.ExcelWriter(OUT_FILE, engine="openpyxl") as writer:
        df_buses_after.to_excel(writer, sheet_name="Buses_After", index=False)
        df_lines.to_excel(writer, sheet_name="Lines", index=False)
        df_trafos.to_excel(writer, sheet_name="Transformers", index=False)
        df_sys.to_excel(writer, sheet_name="System_After", index=False)
        df_eval.to_excel(writer, sheet_name="ObjectiveHistory", index=False)
        if not df_vars.empty:
            df_vars.to_excel(writer, sheet_name="BestSolutionVars", index=False)
        if not df_storage_elems.empty:
            df_storage_elems.to_excel(writer, sheet_name="BestStorageElements", index=False)
        # Write components history and best components
        try:
            df_comp_hist.to_excel(writer, sheet_name="ComponentsHistory", index=False)
        except Exception:
            pass
        if best_components is not None:
            try:
                df_best_comp = pd.DataFrame([best_components])
                df_best_comp.to_excel(writer, sheet_name="BestComponents", index=False)
            except Exception:
                pass

    print(f"Wyniki zapisane do {OUT_FILE}")

    # --- 2D surface scan (optional; controlled by SURFACE_ENABLED flag) ---
    try:
        run_surface_scan(app, vars_def, ldf, res)
    except Exception as _surf_exc:
        print(f"Błąd podczas skanowania powierzchni: {_surf_exc}")
        traceback.print_exc()

    # Optional plots
    try:
        if res.get("best_per_iter"):
            plt.figure(figsize=(8, 5))
            plt.plot(res["best_per_iter"], marker="o", linewidth=2)
            plt.title("Przebieg optymalizacji (funkcja celu)")
            plt.xlabel("Iteracja")
            plt.ylabel("Funkcja celu")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(os.path.dirname(OUT_FILE), "Opt_Funkcja_celu.png"), dpi=300)
            plt.show()
    except Exception:
        pass

    try:
        # Voltage phases comparison plot (before vs after)
        plt.figure(figsize=(10, 5))
        for phase in ["U1 [p.u.]", "U2 [p.u.]", "U3 [p.u.]"]:
            u_before = [b[phase] for b in buses_before if b.get(phase, 0) and b.get(phase, 0) > 0]
            u_after = [b[phase] for b in buses_after if b.get(phase, 0) and b.get(phase, 0) > 0]
            if u_before:
                plt.plot(range(len(u_before)), u_before, marker='o', label=f"{phase} Before")
            if u_after:
                plt.plot(range(len(u_after)), u_after, marker='x', label=f"{phase} After")

        plt.xlabel("Węzeł")
        plt.ylabel("Napięcie [p.u.]")
        plt.title("Porównanie napięć faz węzłów przed i po optymalizacji")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(OUT_FILE), "Voltage_Phases.png"), dpi=300)
        plt.show()
    except Exception:
        pass

    # Clean up PowerFactory object reference
    try:
        app = None
    except Exception:
        pass


if __name__ == "__main__":
    main()