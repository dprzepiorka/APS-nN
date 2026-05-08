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
import time
import math
import random
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

METHOD = "OSA"  # "PSO", "CEO", "PO", "DOA", "RTO", "SPO- długi", "BPB", "KLA", "KEO", "DOE", "OSA", "ECO", "GOA", "COO"
OBJECTIVE = "VoltageTarget"  # "LossP" or "VoltageUnbalance" or "VoltageTarget"

N_ITER = 1000
N_PARTICLES = 100
N = 1
W = 0.7
C1 = 1.5
C2 = 1.5
PENALTY = 1e6
# Multiplier applied when loadflow clearly failed (very large penalty)
LARGE_PENALTY_MULTIPLIER = 1e4

RANDOM_SEED = 42

# Small delay after each PF run (avoid overloading PowerFactory)
EVAL_DELAY = 0.01

# STORAGE defaults (per phase)
STORAGE_P_MIN = -50.0
STORAGE_P_MAX = 50.0
STORAGE_Q_MIN = -0.0001
STORAGE_Q_MAX = 0.0001

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


def load_vars(file_path, sheet="Vars"):
    """
    Load variable definitions from Excel sheet "Vars" (columns: name, pf_class, attr, min, max)
    """
    df = pd.read_excel(file_path, sheet_name=sheet)
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
        df = pd.read_excel(file_path, sheet_name=sheet)
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
    """
    # apply non-storage variables
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

    # handle storage vars if defined
    global STORAGE_CANDIDATES
    if not STORAGE_CANDIDATES:
        return None, None

    # find index variable name 'storage_node_index' in vars_def
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

    # read storage power Q/P variables positions (expected order)
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
        loads_df = pd.read_excel(file_path, sheet_name="Loads")
        gens_df = pd.read_excel(file_path, sheet_name="Generators")
        pv_df = pd.read_excel(file_path, sheet_name="PV")
        ES_df = pd.read_excel(file_path, sheet_name="StatGen")
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
# MAIN
# -------------------------
def main():
    global STORAGE_CANDIDATES, COMPONENT_HISTORY
    # ensure PowerFactory available
    if powerfactory is None:
        print("powerfactory package not available. This script must run in PowerFactory Python environment.")
        return

    app = powerfactory.GetApplicationExt(USER)
    app.ActivateProject(PROJECT_NAME)
    ldf = app.GetFromStudyCase("ComLdf")

    vars_def = load_vars(EXCEL_FILE)
    print(f"Wczytano {len(vars_def)} zmiennych do optymalizacji.")

    # Load storage candidates and append storage variables to vars_def if present
    STORAGE_CANDIDATES = load_storage_candidates(EXCEL_FILE)
    K = len(STORAGE_CANDIDATES)
    if K > 0:
        # add storage variables to vars_def: index + P1,P2,P3,Q1,Q2,Q3
        vars_def.append({
            "name": "storage_node_index",
            "pf_class": "choice",
            "attr": "",
            "min": 0.0,
            "max": float(max(0, K - 1))
        })
        vars_def += [
            {"name": "storage_P1", "pf_class": "storage", "attr": "", "min": STORAGE_P_MIN, "max": STORAGE_P_MAX},
            {"name": "storage_P2", "pf_class": "storage", "attr": "", "min": STORAGE_P_MIN, "max": STORAGE_P_MAX},
            {"name": "storage_P3", "pf_class": "storage", "attr": "", "min": STORAGE_P_MIN, "max": STORAGE_P_MAX},
            {"name": "storage_Q1", "pf_class": "storage", "attr": "", "min": STORAGE_Q_MIN, "max": STORAGE_Q_MAX},
            {"name": "storage_Q2", "pf_class": "storage", "attr": "", "min": STORAGE_Q_MIN, "max": STORAGE_Q_MAX},
            {"name": "storage_Q3", "pf_class": "storage", "attr": "", "min": STORAGE_Q_MIN, "max": STORAGE_Q_MAX},
        ]
        print(f"Dodano zmienne magazynu (kandydatów: {K}) do vars_def.")

    # Set initial elements from Excel sheets
    load_and_set_elements_from_excel(app, EXCEL_FILE)

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

    def obj(x):
        return objective_function(app, vars_def, x, ldf)

    time_start = time.time()
    # Choose optimizer
    res = None
    METHOD_U = METHOD.upper()
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

    time_end = time.time()
    print("running time:{:.5f}".format(time_end - time_start))
    app.PrintError("running time:{:.5f}".format(time_end - time_start))

    # safety: ensure res exists
    if res is None:
        print("No result returned by optimizer.")
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