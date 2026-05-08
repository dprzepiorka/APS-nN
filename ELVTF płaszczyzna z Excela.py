"""
2D surface scan driven directly from an Excel worksheet.

Workbook: dane.xlsx
Worksheet: Surface2D

Expected columns:
    name       - PowerFactory element name
    pf_class   - optional, e.g. ElmGenstat / ElmSym / ElmPvsys / ElmLod / ElmTr2 / ElmLne / ElmTerm
    attr       - PowerFactory attribute to control, e.g. pgini / qgini / nntap
    regulates  - what this variable contributes to axis sum:
                 * P_STORAGE      -> contributes to X (sum active power of storages)
                 * Q_PV_STORAGE   -> contributes to Y (sum reactive power of PV + storages)
                 (column aliases accepted: "co reguluje", "group")
    opt_value  - optimum / center value of the plane
    min        - lower bound for the plane axis
    max        - upper bound for the plane axis

The script:
1. loads the base network state from dane.xlsx sheets Loads/Generators/PV/StatGen,
2. loads variable definitions from Surface2D,
3. builds aggregate axes:
   X = ΣP magazynów energii,
   Y = ΣQ źródeł PV i magazynów,
4. redistributes each target sum onto variables from Surface2D within their [min,max],
5. evaluates the existing objective function (without penalty) in each grid point,
5. saves Excel + NPZ + 3D plot + heatmap.
"""

import os
import sys
import time
import shutil
import tempfile
import subprocess
import traceback

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reuse the existing project objective and snapshot logic
import ELVTF as core

# PowerFactory python path - adjust if necessary for your installation
sys.path.append(r"C:\Program Files\DIgSILENT\PowerFactory 2024 SP5A\Python\3.12")
try:
    import powerfactory
except Exception:
    powerfactory = None
    print("Warning: powerfactory module not available.")


# -------------------------
# CONFIG
# -------------------------
EXCEL_FILE = r"N:\ksiz\STUDIA_DOKTORANCKIE\PRZEPIORKA\Modele sieci\Niskie napiecie\APS\Optymalizacja\dane.xlsx"
SURFACE_SHEET = "Surface2D"
PROJECT_NAME = "ELVTF3x1F"
USER = "minik"

GRID_NX = 50
GRID_NY = 50
PROGRESS_EVERY = 25  # co ile punktów wypisywać postęp (0/None = auto)

OUT_SURFACE_FILE = r"N:\ksiz\STUDIA_DOKTORANCKIE\PRZEPIORKA\Modele sieci\Niskie napiecie\APS\Optymalizacja\surface_scan_excel_axes.xlsx"
OUT_DIR = os.path.dirname(OUT_SURFACE_FILE)

# Keep the same objective settings as the main project file
core.EVAL_DELAY = 0.0
core.STORAGE_CANDIDATES = []
core.COMPONENT_HISTORY = []

COMMON_PF_CLASSES = [
    "ElmGenstat",
    "ElmSym",
    "ElmPvsys",
    "ElmLod",
    "ElmTr2",
    "ElmLne",
    "ElmTerm",
]

GROUP_P_STORAGE = {
    "P_STORAGE", "PSTORAGE", "P_MAGAZYNOW", "P_MAGAZYNY", "X"
}
GROUP_Q_PV_STORAGE = {
    "Q_PV_STORAGE", "QPVSTORAGE", "Q_PV_MAGAZYNOW", "Q_PV_MAGAZYNY", "Y"
}

X_LABEL = "ΣP magazynów energii"
Y_LABEL = "ΣQ PV + magazyny"


# -------------------------
# Excel helpers
# -------------------------
def read_excel_safe(file_path, sheet_name=None, **kwargs):
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
            try:
                shutil.copyfile(path, tmp_path)
                copied = True
            except Exception:
                copied = False

            if not copied:
                cmd = f'copy /Y "{path}" "{tmp_path}" >nul'
                res = subprocess.run(["cmd", "/c", cmd], capture_output=True, text=True)
                copied = res.returncode == 0 and os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0

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
                copied = res.returncode == 0 and os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0

            if not copied:
                raise RuntimeError(f"Nie udało się skopiować pliku Excel do pliku tymczasowego: {path}")

            return pd.read_excel(tmp_path, sheet_name=sheet_name, engine="openpyxl", **kwargs)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass


def load_and_set_elements_from_excel_safe(app, file_path):
    try:
        loads_df = read_excel_safe(file_path, sheet_name="Loads")
        gens_df = read_excel_safe(file_path, sheet_name="Generators")
        pv_df = read_excel_safe(file_path, sheet_name="PV")
        es_df = read_excel_safe(file_path, sheet_name="StatGen")
    except Exception as e:
        print(f"Błąd przy wczytywaniu danych bazowych z {file_path}: {e}")
        return

    for _, row in loads_df.iterrows():
        name = str(row["name"]).strip()
        elm = core.find_element(app, name, "ElmLod")
        if not elm:
            continue
        for attr, col in [
            ("plinir", "P1"), ("plinis", "P2"), ("plinit", "P3"),
            ("qlinir", "Q1"), ("qlinis", "Q2"), ("qlinit", "Q3"),
        ]:
            try:
                core._set_element_attr_safe(elm, attr, float(row[col]))
            except Exception:
                pass

    for df, pf_class in [(gens_df, "ElmSym"), (pv_df, "ElmPvsys"), (es_df, "ElmGenstat")]:
        for _, row in df.iterrows():
            name = str(row["name"]).strip()
            elm = core.find_element(app, name, pf_class)
            if not elm:
                continue
            try:
                core._set_element_attr_safe(elm, "pgini", float(row["P"]))
                core._set_element_attr_safe(elm, "qgini", float(row["Q"]))
            except Exception:
                pass

    print("Parametry Loads, PV, ES i Generators zostały wczytane i ustawione w PowerFactory.")


# -------------------------
# Surface definition
# -------------------------
def resolve_pf_class(app, name, pf_class):
    pf_class = "" if pf_class is None else str(pf_class).strip()
    if pf_class and pf_class.lower() != "nan":
        elm = core.find_element(app, name, pf_class)
        if elm is not None:
            return pf_class

    for candidate_class in COMMON_PF_CLASSES:
        elm = core.find_element(app, name, candidate_class)
        if elm is not None:
            return candidate_class

    return None


def normalize_surface_columns(df):
    mapped = {}
    aliases = {
        "co reguluje": "regulates",
        "co_reguluje": "regulates",
        "coreguluje": "regulates",
        "reguluje": "regulates",
        "group": "regulates",
        "axis_group": "regulates",
    }
    for col in df.columns:
        key = str(col).strip().lower()
        mapped[col] = aliases.get(key, key)
    return df.rename(columns=mapped)


def normalize_group_name(value):
    txt = str(value).strip().upper().replace(" ", "")
    if txt in GROUP_P_STORAGE:
        return "P_STORAGE"
    if txt in GROUP_Q_PV_STORAGE:
        return "Q_PV_STORAGE"
    return txt


def load_surface_definition(app, file_path, sheet_name):
    df = read_excel_safe(file_path, sheet_name=sheet_name)
    df = normalize_surface_columns(df)
    df = df.dropna(how="all")

    required = ["name", "attr", "opt_value", "min", "max"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Brak kolumn w arkuszu {sheet_name}: {missing}")

    has_regulates = "regulates" in df.columns
    has_axis = "axis" in df.columns

    if not has_regulates and not has_axis:
        print("Uwaga: brak kolumny 'regulates' i 'axis' w Surface2D. Spróbuję przypisać grupy na podstawie attr (p* -> P_STORAGE, q* -> Q_PV_STORAGE).")

    rows = []
    for _, row in df.iterrows():
        name = str(row["name"]).strip()
        if not name or name.lower() == "nan":
            continue

        attr = str(row["attr"]).strip()
        if not attr or attr.lower() == "nan":
            continue

        if has_regulates:
            group = normalize_group_name(row["regulates"])
        elif has_axis:
            group = normalize_group_name(row["axis"])
        else:
            attr_l = attr.lower()
            if attr_l.startswith("p"):
                group = "P_STORAGE"
            elif attr_l.startswith("q"):
                group = "Q_PV_STORAGE"
            else:
                group = ""

        if group not in ("P_STORAGE", "Q_PV_STORAGE"):
            continue

        pf_class = resolve_pf_class(app, name, row.get("pf_class", ""))
        if pf_class is None:
            raise ValueError(f"Nie znaleziono elementu '{name}' w modelu PowerFactory.")

        opt_value = float(row["opt_value"])
        vmin = float(row["min"])
        vmax = float(row["max"])
        if vmin > vmax:
            vmin, vmax = vmax, vmin

        rows.append({
            "group": group,
            "name": name,
            "pf_class": pf_class,
            "attr": attr,
            "opt_value": opt_value,
            "min": vmin,
            "max": vmax,
            "label": f"{name}.{attr}",
        })

    if not rows:
        raise ValueError(f"Arkusz {sheet_name} nie zawiera poprawnych wierszy sterujących.")

    if not any(r["group"] == "P_STORAGE" for r in rows):
        raise ValueError(f"Arkusz {sheet_name} musi zawierać co najmniej jeden wiersz z regulates=P_STORAGE.")

    if not any(r["group"] == "Q_PV_STORAGE" for r in rows):
        raise ValueError(f"Arkusz {sheet_name} musi zawierać co najmniej jeden wiersz z regulates=Q_PV_STORAGE.")

    return rows


# -------------------------
# Plot helpers
# -------------------------
def plot_surface_3d(x_arr, y_arr, j_surface, x_label, y_label, j_star, out_dir):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    xx, yy = np.meshgrid(x_arr, y_arr)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(xx, yy, j_surface, cmap="viridis", alpha=0.85, edgecolor="none")
    ax.scatter([x_arr[len(x_arr)//2]], [y_arr[len(y_arr)//2]], [j_star], color="red", s=100, zorder=6, label="Punkt optymalny")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel("J (funkcja celu)")
    ax.set_title("Powierzchnia funkcji celu 3D")
    fig.colorbar(surf, ax=ax, shrink=0.45, label="J")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, "Surface_ExcelAxes_3D.png")
    plt.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Wykres 3D zapisany: {path}")


def plot_surface_heatmap(x_arr, y_arr, j_surface, x_label, y_label, out_dir):
    xx, yy = np.meshgrid(x_arr, y_arr)
    fig, ax = plt.subplots(figsize=(8, 6))
    cf = ax.contourf(xx, yy, j_surface, levels=30, cmap="viridis")
    ax.contour(xx, yy, j_surface, levels=30, colors="k", linewidths=0.3, alpha=0.4)
    plt.colorbar(cf, ax=ax, label="J")
    ax.scatter([x_arr[len(x_arr)//2]], [y_arr[len(y_arr)//2]], color="red", s=100, zorder=6, label="Punkt optymalny")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title("Mapa ciepła funkcji celu")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, "Surface_ExcelAxes_Heatmap.png")
    plt.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Mapa ciepła zapisana: {path}")


def objective_no_penalty(app, vars_def, x, ldf):
    """
    Evaluate the existing objective pipeline but return value WITHOUT penalty term.

    Uses core.objective_function to keep identical PF execution path and then extracts
    the latest component record from core.COMPONENT_HISTORY:
      [C1, C2, C3, C4, C5, penalty, total]
    Returned value = weighted sum(C1..C5), i.e. no penalty.
    """
    _ = core.objective_function(app, vars_def, x, ldf)

    if not core.COMPONENT_HISTORY:
        return np.nan

    last = core.COMPONENT_HISTORY[-1]
    if len(last) < 7:
        return np.nan

    c1, c2, c3, c4, c5, _, _ = last
    comps = np.array([c1, c2, c3, c4, c5], dtype=float)
    if np.any(np.isnan(comps)) or np.any(np.isinf(comps)):
        return np.nan

    w = np.array(getattr(core, "WEIGHTS", [0.2] * 5), dtype=float)
    if len(w) != 5:
        w = np.array([0.2] * 5, dtype=float)

    return float(np.dot(w, comps))


# -------------------------
# Main 2D scan
# -------------------------
def build_centered_axis(vmin, vmax, vopt, n_points, axis_name):
    if n_points < 3:
        n_points = 3
    if n_points % 2 == 0:
        n_points += 1
        print(f"Uwaga: {axis_name} miała parzystą liczbę punktów, zwiększam do {n_points}, aby optimum było dokładnie w środku.")

    if vopt < vmin:
        print(f"Uwaga: opt_value dla osi {axis_name} < min, ustawiam opt_value=min.")
        vopt = vmin
    elif vopt > vmax:
        print(f"Uwaga: opt_value dla osi {axis_name} > max, ustawiam opt_value=max.")
        vopt = vmax

    mid = n_points // 2
    left = np.linspace(vmin, vopt, mid + 1)[:-1]
    right = np.linspace(vopt, vmax, n_points - mid)
    return np.concatenate([left, right])


def redistribute_to_target(base, lb, ub, target, eps=1e-9, max_iter=50):
    base = np.array(base, dtype=float)
    lb = np.array(lb, dtype=float)
    ub = np.array(ub, dtype=float)

    vals = np.clip(base.copy(), lb, ub)
    target = float(np.clip(target, np.sum(lb), np.sum(ub)))

    for _ in range(max_iter):
        rem = target - float(np.sum(vals))
        if abs(rem) <= 1e-10:
            break

        if rem > 0:
            free = np.where(vals < ub - 1e-12)[0]
            if len(free) == 0:
                break
            capacity = ub[free] - vals[free]
        else:
            free = np.where(vals > lb + 1e-12)[0]
            if len(free) == 0:
                break
            capacity = vals[free] - lb[free]

        w = np.abs(base[free]) + eps
        w_sum = float(np.sum(w))
        if w_sum <= 0:
            w = np.ones(len(free), dtype=float) / max(1, len(free))
        else:
            w = w / w_sum

        proposal = rem * w
        step = np.sign(rem) * np.minimum(np.abs(proposal), capacity)
        vals[free] = vals[free] + step

    return np.clip(vals, lb, ub)


def run_explicit_surface_scan(app, ldf, vars_def, x_star, x_axis_meta, y_axis_meta, p_indices, q_indices):
    x_arr = build_centered_axis(x_axis_meta["min"], x_axis_meta["max"], x_axis_meta["opt"], GRID_NX, "X")
    y_arr = build_centered_axis(y_axis_meta["min"], y_axis_meta["max"], y_axis_meta["opt"], GRID_NY, "Y")
    nx = len(x_arr)
    ny = len(y_arr)

    j_surface = np.full((ny, nx), np.nan)

    total_pts = nx * ny
    done = 0
    t0 = time.time()
    print_every = int(PROGRESS_EVERY) if PROGRESS_EVERY else max(1, total_pts // 20)
    print_every = max(1, print_every)

    print("=" * 70)
    print("=== Skanowanie 2D po osiach sumarycznych ===")
    print(f"X: {X_LABEL}  ∈ [{x_axis_meta['min']}, {x_axis_meta['max']}]  ({nx} punktów, optimum w środku)")
    print(f"Y: {Y_LABEL}  ∈ [{y_axis_meta['min']}, {y_axis_meta['max']}]  ({ny} punktów, optimum w środku)")
    print(f"Do policzenia łącznie: {total_pts} punktów")
    print(f"Raport postępu co: {print_every} punkt(ów)")
    print("=" * 70)

    j_star = objective_no_penalty(app, vars_def, x_star, ldf)

    p_lb = np.array([vars_def[i]["min"] for i in p_indices], dtype=float)
    p_ub = np.array([vars_def[i]["max"] for i in p_indices], dtype=float)
    p_base = np.array([x_star[i] for i in p_indices], dtype=float)

    q_lb = np.array([vars_def[i]["min"] for i in q_indices], dtype=float)
    q_ub = np.array([vars_def[i]["max"] for i in q_indices], dtype=float)
    q_base = np.array([x_star[i] for i in q_indices], dtype=float)

    for iy, y_val in enumerate(y_arr):
        for ix, x_val in enumerate(x_arr):
            x_scan = np.array(x_star, dtype=float)

            p_vals = redistribute_to_target(p_base, p_lb, p_ub, float(x_val))
            for k, idx_var in enumerate(p_indices):
                x_scan[idx_var] = p_vals[k]

            q_vals = redistribute_to_target(q_base, q_lb, q_ub, float(y_val))
            for k, idx_var in enumerate(q_indices):
                x_scan[idx_var] = q_vals[k]

            try:
                j_val = objective_no_penalty(app, vars_def, x_scan, ldf)
            except Exception as e:
                j_val = np.nan
                print(f"  [scan iy={iy},ix={ix}] Wyjątek: {e}")

            j_surface[iy, ix] = j_val
            done += 1

            if done % print_every == 0 or done == total_pts:
                elapsed = time.time() - t0
                eta = (elapsed / done) * (total_pts - done) if done > 0 else 0.0
                left = total_pts - done
                print(
                    f"  Punkt #{done}/{total_pts} (Y:{iy+1}/{ny}, X:{ix+1}/{nx}) | "
                    f"X={x_val:.6g}, Y={y_val:.6g} | zostało={left} | "
                    f"{100.0 * done / total_pts:.0f}% | czas={elapsed:.1f}s | ETA={eta:.1f}s | "
                    f"J_min_dotąd={np.nanmin(j_surface):.5g}"
                )

    # restore optimum
    try:
        objective_no_penalty(app, vars_def, x_star, ldf)
        print("Model przywrócony do punktu optymalnego po skanowaniu.")
    except Exception as e:
        print(f"Uwaga: błąd przywracania optimum po skanowaniu: {e}")

    out_dir = os.path.dirname(OUT_SURFACE_FILE)

    try:
        df_x = pd.DataFrame({"ix": range(nx), "x_value": x_arr})
        df_y = pd.DataFrame({"iy": range(ny), "y_value": y_arr})
        df_j = pd.DataFrame(
            j_surface,
            index=[f"Y={v:.6g}" for v in y_arr],
            columns=[f"X={v:.6g}" for v in x_arr],
        )
        df_def = pd.DataFrame(vars_def)
        df_meta = pd.DataFrame([
            {"key": "date", "value": str(pd.Timestamp.now())},
            {"key": "grid_nx", "value": nx},
            {"key": "grid_ny", "value": ny},
            {"key": "x_label", "value": X_LABEL},
            {"key": "y_label", "value": Y_LABEL},
            {"key": "x_opt_sum", "value": x_axis_meta["opt"]},
            {"key": "y_opt_sum", "value": y_axis_meta["opt"]},
            {"key": "x_center_index", "value": nx // 2},
            {"key": "y_center_index", "value": ny // 2},
            {"key": "x_center_value", "value": x_arr[nx // 2]},
            {"key": "y_center_value", "value": y_arr[ny // 2]},
            {"key": "j_star", "value": j_star},
            {"key": "j_min_scan", "value": float(np.nanmin(j_surface)) if not np.all(np.isnan(j_surface)) else np.nan},
        ])

        with pd.ExcelWriter(OUT_SURFACE_FILE, engine="openpyxl") as writer:
            df_def.to_excel(writer, sheet_name="Definition", index=False)
            df_x.to_excel(writer, sheet_name="Axis_X", index=False)
            df_y.to_excel(writer, sheet_name="Axis_Y", index=False)
            df_j.to_excel(writer, sheet_name="Surface_J", index=True)
            df_meta.to_excel(writer, sheet_name="Meta", index=False)
        print(f"Wyniki powierzchni zapisane: {OUT_SURFACE_FILE}")

        npz_path = os.path.splitext(OUT_SURFACE_FILE)[0] + ".npz"
        np.savez(npz_path, x=x_arr, y=y_arr, J=j_surface, j_star=np.array([j_star]))
        print(f"NPZ zapisany: {npz_path}")
    except Exception as e:
        print(f"Błąd zapisu wyników powierzchni: {e}")
        traceback.print_exc()

    plot_surface_3d(x_arr, y_arr, j_surface, X_LABEL, Y_LABEL, j_star, out_dir)
    plot_surface_heatmap(x_arr, y_arr, j_surface, X_LABEL, Y_LABEL, out_dir)

    return x_arr, y_arr, j_surface


def main():
    if powerfactory is None:
        print("powerfactory package not available. This script must run in PowerFactory Python environment.")
        return

    app = powerfactory.GetApplicationExt(USER)
    if app is None:
        print("Nie można połączyć z PowerFactory.")
        return

    app.ActivateProject(PROJECT_NAME)
    ldf = app.GetFromStudyCase("ComLdf")
    if ldf is None:
        print("Nie znaleziono ComLdf w Study Case.")
        return

    print("Wczytuję dane bazowe z Excela...")
    load_and_set_elements_from_excel_safe(app, EXCEL_FILE)

    print(f"Wczytuję definicję płaszczyzny z arkusza '{SURFACE_SHEET}'...")
    surface_rows = load_surface_definition(app, EXCEL_FILE, SURFACE_SHEET)

    vars_def = []
    x_star_vals = []
    p_indices = []
    q_indices = []
    index_by_key = {}

    for row in surface_rows:
        key = (row["name"], row["pf_class"], row["attr"])
        if key in index_by_key:
            idx = index_by_key[key]
        else:
            idx = len(vars_def)
            index_by_key[key] = idx
            vars_def.append({
                "name": row["name"],
                "pf_class": row["pf_class"],
                "attr": row["attr"],
                "min": row["min"],
                "max": row["max"],
            })
            x_star_vals.append(float(row["opt_value"]))

        if row["group"] == "P_STORAGE":
            p_indices.append(idx)
        elif row["group"] == "Q_PV_STORAGE":
            q_indices.append(idx)

    # unique group index lists (preserve order)
    p_indices = list(dict.fromkeys(p_indices))
    q_indices = list(dict.fromkeys(q_indices))

    x_star = np.array(x_star_vals, dtype=float)

    x_axis_meta = {
        "opt": float(np.sum([x_star[i] for i in p_indices])),
        "min": float(np.sum([vars_def[i]["min"] for i in p_indices])),
        "max": float(np.sum([vars_def[i]["max"] for i in p_indices])),
    }
    y_axis_meta = {
        "opt": float(np.sum([x_star[i] for i in q_indices])),
        "min": float(np.sum([vars_def[i]["min"] for i in q_indices])),
        "max": float(np.sum([vars_def[i]["max"] for i in q_indices])),
    }

    print("Punkt centralny płaszczyzny (sumaryczny):")
    print(f"  X ({X_LABEL}) = {x_axis_meta['opt']}")
    print(f"  Y ({Y_LABEL}) = {y_axis_meta['opt']}")

    try:
        objective_no_penalty(app, vars_def, x_star, ldf)
        buses, lines, trafos, sys_info = core.collect_results_snapshot(app)
        start_file = os.path.splitext(OUT_SURFACE_FILE)[0] + "_start.xlsx"
        with pd.ExcelWriter(start_file, engine="openpyxl") as writer:
            pd.DataFrame(buses).to_excel(writer, sheet_name="Buses", index=False)
            pd.DataFrame(lines).to_excel(writer, sheet_name="Lines", index=False)
            pd.DataFrame(trafos).to_excel(writer, sheet_name="Transformers", index=False)
            pd.DataFrame(sys_info).to_excel(writer, sheet_name="System", index=False)
        print(f"Snapshot punktu centralnego zapisany: {start_file}")
    except Exception as e:
        print(f"Uwaga: nie udało się zapisać snapshotu punktu centralnego: {e}")

    run_explicit_surface_scan(app, ldf, vars_def, x_star, x_axis_meta, y_axis_meta, p_indices, q_indices)


if __name__ == "__main__":
    main()
