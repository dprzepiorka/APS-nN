"""
Puma Optimizer (PO) - Python implementation translated from PO.m

Interface similar to CEO/PSO classes in this project:

Usage:
    po = PO(func, n_sol, dim, lb, ub, max_iter,
            autosave_every_iters=0, autosave_path="po_checkpoint.npz", eval_delay=0.0, rng_seed=None)
    res = po.optimize()
    res -> {"gbest": array, "gbest_val": float, "best_per_iter": list}

Notes:
- func(x) should accept 1D numpy array and return scalar objective.
- This implementation follows the structure of the Matlab PO.m (Exploration/Exploitation phases).
- Adds safe evaluation (exceptions -> +inf), optional eval_delay, autosave and checkpointing.
"""
import numpy as np
import time
import os
import traceback

class PO:
    def __init__(self, func, n_sol, dim, lb, ub, max_iter,
                 autosave_every_iters=0, autosave_path="po_checkpoint.npz", eval_delay=0.0, rng_seed=None):
        self.func = func
        self.n_sol = int(n_sol)
        self.dim = int(dim)
        self.lb = np.array(lb, dtype=float)
        self.ub = np.array(ub, dtype=float)
        self.max_iter = int(max_iter)

        self.autosave_every_iters = int(autosave_every_iters)
        self.autosave_path = autosave_path
        self.eval_delay = float(eval_delay)

        if rng_seed is not None:
            np.random.seed(rng_seed)

        # bookkeeping
        self.best_per_iter = []
        self.iter = 0

        # initialize population (Sol as list of dicts {'X':..., 'Cost':...})
        self.Sol = []
        for i in range(self.n_sol):
            x = np.random.uniform(self.lb, self.ub, self.dim)
            c = self._safe_eval(x)
            self.Sol.append({'X': x, 'Cost': c})
        # find best
        costs = [s['Cost'] for s in self.Sol]
        idx = int(np.argmin(costs))
        self.Best = self.Sol[idx].copy()
        self.Initial_Best = self.Best.copy()

        # helper vars (translate from MATLAB)
        self.UnSelected = [1, 1]
        self.F3_Explore = 0.0
        self.F3_Exploit = 0.0
        self.Seq_Time_Explore = [1.0, 1.0, 1.0]
        self.Seq_Time_Exploit = [1.0, 1.0, 1.0]
        self.Seq_Cost_Explore = [1.0, 1.0, 1.0]
        self.Seq_Cost_Exploit = [1.0, 1.0, 1.0]
        self.Score_Explore = 0.0
        self.Score_Exploit = 0.0
        self.PF = [0.5, 0.5, 0.3]
        self.PF_F3 = []
        self.Mega_Explor = 0.99
        self.Mega_Exploit = 0.99
        self.Flag_Change = 1

    def _safe_eval(self, x):
        try:
            v = float(self.func(np.array(x, dtype=float)))
        except Exception as e:
            try:
                with open("failed_evals_po.csv", "a", encoding="utf-8") as f:
                    f.write(f"{time.time()},{','.join(map(str, x))},exception,{str(e)}\n")
            except Exception:
                pass
            v = float("inf")
        if self.eval_delay:
            time.sleep(self.eval_delay)
        return v

    def save_checkpoint(self, path=None):
        if path is None:
            path = self.autosave_path
        try:
            np.savez(path,
                     Sol_X=np.array([s['X'] for s in self.Sol]),
                     Sol_C=np.array([s['Cost'] for s in self.Sol]),
                     Best_X=self.Best['X'],
                     Best_C=self.Best['Cost'],
                     iter=self.iter,
                     best_per_iter=np.array(self.best_per_iter))
        except Exception as e:
            try:
                with open("po_save_error.log", "a", encoding="utf-8") as f:
                    f.write(f"{time.time()} save error: {e}\n")
                    f.write(traceback.format_exc())
            except Exception:
                pass

    # ---------------------
    # Exploration phase (translated)
    # ---------------------
    def Exploration(self, Sol):
        # sort by cost
        Sol_sorted = sorted(Sol, key=lambda s: s['Cost'])
        pCR = 0.20
        PCR = 1 - pCR
        p = PCR / self.n_sol

        NewSol = []
        for i in range(self.n_sol):
            x = Sol_sorted[i]['X']
            A = np.random.permutation(self.n_sol)
            A = A[A != i]  # remove self index
            # ensure enough indices
            # pad with random selections if needed
            if A.size < 6:
                extra = np.random.choice(self.n_sol, 6 - A.size, replace=True)
                A = np.concatenate([A, extra])
            a, b, c, d, e, f = A[:6]
            G = 2 * np.random.rand() - 1
            if np.random.rand() < 0.5:
                y = np.random.rand(self.dim) * (self.ub - self.lb) + self.lb
            else:
                y = (Sol_sorted[a]['X'] + G * (Sol_sorted[a]['X'] - Sol_sorted[b]['X'])
                     + G * (((Sol_sorted[a]['X'] - Sol_sorted[b]['X']) - (Sol_sorted[c]['X'] - Sol_sorted[d]['X']))
                            + ((Sol_sorted[c]['X'] - Sol_sorted[d]['X']) - (Sol_sorted[e]['X'] - Sol_sorted[f]['X']))))
            y = np.maximum(y, self.lb)
            y = np.minimum(y, self.ub)
            z = np.zeros_like(x)
            j0 = np.random.randint(0, x.size)
            for j in range(x.size):
                if j == j0 or np.random.rand() <= pCR:
                    z[j] = y[j]
                else:
                    z[j] = x[j]
            new_x = z
            new_cost = self._safe_eval(new_x)
            if new_cost < Sol_sorted[i]['Cost']:
                NewSol.append({'X': new_x, 'Cost': new_cost})
            else:
                # increase pCR (PCR adaptation)
                pCR = pCR + p
                NewSol.append(Sol_sorted[i])
        return NewSol

    # ---------------------
    # Exploitation phase (translated)
    # ---------------------
    def Exploitation(self, Sol, Best, Iter):
        Q = 0.67
        Beta = 2.0
        NewSol = []
        Xmat = np.vstack([s['X'] for s in Sol])
        mbest = np.mean(Xmat, axis=0)  # average position
        for i in range(self.n_sol):
            beta1 = 2 * np.random.rand()
            beta2 = np.random.randn(self.dim)
            w = np.random.randn(self.dim)
            v = np.random.randn(self.dim)
            F1 = np.random.randn(self.dim) * np.exp(2 - Iter * (2 / max(1, self.max_iter)))
            F2 = w * (v ** 2) * np.cos((2 * np.random.rand()) * w)
            R_1 = 2 * np.random.rand() - 1
            S1 = (2 * np.random.rand() - 1) + np.random.randn(self.dim)
            S2 = (F1 * R_1 * Sol[i]['X'] + F2 * (1 - R_1) * Best['X'])
            # avoid division by zero by adding small eps
            VEC = S2 / (S1 + np.finfo(float).eps)
            if np.random.rand() <= 0.5:
                Xatack = VEC
                if np.random.rand() > Q:
                    new_x = Best['X'] + beta1 * np.exp(beta2) * (Sol[np.random.randint(self.n_sol)]['X'] - Sol[i]['X'])
                else:
                    new_x = beta1 * Xatack - Best['X']
            else:
                r1 = np.random.randint(0, self.n_sol)
                new_x = (mbest * Sol[r1]['X'] - ((-1) ** np.random.randint(0, 2)) * Sol[i]['X']) / (1 + (Beta * np.random.rand()))
            new_x = np.maximum(new_x, self.lb)
            new_x = np.minimum(new_x, self.ub)
            new_cost = self._safe_eval(new_x)
            if new_cost < Sol[i]['Cost']:
                NewSol.append({'X': new_x, 'Cost': new_cost})
            else:
                NewSol.append(Sol[i])
        return NewSol

    # ---------------------
    # Boundary check util (vectorized)
    # ---------------------
    def BoundaryCheck(self, X):
        X = np.array(X, dtype=float)
        X = np.minimum(X, self.ub)
        X = np.maximum(X, self.lb)
        return X

    # ---------------------
    # Main optimize method
    # ---------------------
    def optimize(self):
        try:
            # Unexperienced phase (first 3 iterations)
            Costs_Explor = []
            Costs_Exploit = []
            for Iter in range(1, 4):
                Sol_Explor = self.Exploration(self.Sol)
                Costs_Explor.append(min([s['Cost'] for s in Sol_Explor]))
                Sol_Exploit = self.Exploitation(self.Sol, self.Best, self.max_iter)
                Costs_Exploit.append(min([s['Cost'] for s in Sol_Exploit]))

                # merge and select best nSol
                merged = self.Sol + Sol_Explor + Sol_Exploit
                merged_sorted = sorted(merged, key=lambda s: s['Cost'])
                self.Sol = merged_sorted[:self.n_sol]
                # update best
                self.Best = self.Sol[0].copy()
                self.iter = Iter
                self.best_per_iter.append(self.Best['Cost'])
                print(f"Iteration: {Iter} Best Cost = {self.Best['Cost']}")
            # Hyper Initialization
            try:
                self.Seq_Cost_Explore[0] = abs(self.Initial_Best['Cost'] - Costs_Explor[0])
                self.Seq_Cost_Exploit[0] = abs(self.Initial_Best['Cost'] - Costs_Exploit[0])
                self.Seq_Cost_Explore[1] = abs(Costs_Explor[1] - Costs_Explor[0])
                self.Seq_Cost_Exploit[1] = abs(Costs_Exploit[1] - Costs_Exploit[0])
                self.Seq_Cost_Explore[2] = abs(Costs_Explor[2] - Costs_Explor[1])
                self.Seq_Cost_Exploit[2] = abs(Costs_Exploit[2] - Costs_Exploit[1])
            except Exception:
                pass

            # collect PF_F3 if non-zero
            for i in range(3):
                if self.Seq_Cost_Explore[i] != 0:
                    self.PF_F3.append(self.Seq_Cost_Explore[i])
                if self.Seq_Cost_Exploit[i] != 0:
                    self.PF_F3.append(self.Seq_Cost_Exploit[i])

            # compute F1/F2 initial scores
            F1_Explor = self.PF[0] * (self.Seq_Cost_Explore[0] / self.Seq_Time_Explore[0])
            F1_Exploit = self.PF[0] * (self.Seq_Cost_Exploit[0] / self.Seq_Time_Exploit[0])
            F2_Explor = self.PF[1] * (sum(self.Seq_Cost_Explore) / sum(self.Seq_Time_Explore))
            F2_Exploit = self.PF[1] * (sum(self.Seq_Cost_Exploit) / sum(self.Seq_Time_Exploit))
            self.Score_Explore = (self.PF[0] * F1_Explor) + (self.PF[1] * F2_Explor)
            self.Score_Exploit = (self.PF[0] * F1_Exploit) + (self.PF[1] * F2_Exploit)

            # Experienced Phase
            for Iter in range(4, self.max_iter + 1):
                self.iter = Iter
                if self.Score_Explore > self.Score_Exploit:
                    # Exploration
                    SelectFlag = 1
                    self.Sol = self.Exploration(self.Sol)
                    Count_select = self.UnSelected.copy()
                    self.UnSelected[1] += 1
                    self.UnSelected[0] = 1
                    self.F3_Explore = self.PF[2]
                    self.F3_Exploit = self.F3_Exploit + self.PF[2]
                    # update best from Sol
                    TBest = min(self.Sol, key=lambda s: s['Cost'])
                    self.Seq_Cost_Explore[2] = self.Seq_Cost_Explore[1]
                    self.Seq_Cost_Explore[1] = self.Seq_Cost_Explore[0]
                    self.Seq_Cost_Explore[0] = abs(self.Best['Cost'] - TBest['Cost'])
                    if self.Seq_Cost_Explore[0] != 0:
                        self.PF_F3.append(self.Seq_Cost_Explore[0])
                    if TBest['Cost'] < self.Best['Cost']:
                        self.Best = TBest.copy()
                else:
                    # Exploitation
                    SelectFlag = 2
                    self.Sol = self.Exploitation(self.Sol, self.Best, Iter)
                    Count_select = self.UnSelected.copy()
                    self.UnSelected[0] += 1
                    self.UnSelected[1] = 1
                    self.F3_Explore = self.F3_Explore + self.PF[2]
                    self.F3_Exploit = self.PF[2]
                    TBest = min(self.Sol, key=lambda s: s['Cost'])
                    self.Seq_Cost_Exploit[2] = self.Seq_Cost_Exploit[1]
                    self.Seq_Cost_Exploit[1] = self.Seq_Cost_Exploit[0]
                    self.Seq_Cost_Exploit[0] = abs(self.Best['Cost'] - TBest['Cost'])
                    if self.Seq_Cost_Exploit[0] != 0:
                        self.PF_F3.append(self.Seq_Cost_Exploit[0])
                    if TBest['Cost'] < self.Best['Cost']:
                        self.Best = TBest.copy()

                if self.Flag_Change != SelectFlag:
                    self.Flag_Change = SelectFlag
                    self.Seq_Time_Explore[2] = self.Seq_Time_Explore[1]
                    self.Seq_Time_Explore[1] = self.Seq_Time_Explore[0]
                    self.Seq_Time_Explore[0] = Count_select[0]
                    self.Seq_Time_Exploit[2] = self.Seq_Time_Exploit[1]
                    self.Seq_Time_Exploit[1] = self.Seq_Time_Exploit[0]
                    self.Seq_Time_Exploit[0] = Count_select[1]

                # Hyper Initilization (update F1/F2)
                F1_Explor = self.PF[0] * (self.Seq_Cost_Explore[0] / max(self.Seq_Time_Explore[0], 1e-9))
                F1_Exploit = self.PF[0] * (self.Seq_Cost_Exploit[0] / max(self.Seq_Time_Exploit[0], 1e-9))
                F2_Explor = self.PF[1] * (sum(self.Seq_Cost_Explore) / max(sum(self.Seq_Time_Explore), 1e-9))
                F2_Exploit = self.PF[1] * (sum(self.Seq_Cost_Exploit) / max(sum(self.Seq_Time_Exploit), 1e-9))

                # update Mega and Score as in MATLAB
                if self.Score_Explore < self.Score_Exploit:
                    self.Mega_Explor = max((self.Mega_Explor - 0.01), 0.01)
                    self.Mega_Exploit = 0.99
                elif self.Score_Explore > self.Score_Exploit:
                    self.Mega_Explor = 0.99
                    self.Mega_Exploit = max((self.Mega_Exploit - 0.01), 0.01)

                lmn_Explore = 1.0 - self.Mega_Explor
                lmn_Exploit = 1.0 - self.Mega_Exploit

                # protect min(PF_F3)
                min_pf_f3 = min(self.PF_F3) if len(self.PF_F3) > 0 else 1.0

                self.Score_Explore = (self.Mega_Explor * F1_Explor) + (self.Mega_Explor * F2_Explor) + (lmn_Explore * (min_pf_f3 * self.F3_Explore))
                self.Score_Exploit = (self.Mega_Exploit * F1_Exploit) + (self.Mega_Exploit * F2_Exploit) + (lmn_Exploit * (min_pf_f3 * self.F3_Exploit))

                self.best_per_iter.append(self.Best['Cost'])
                self.iter = Iter
                print(f"Iteration: {Iter} Best Cost = {self.Best['Cost']}")
                # autosave
                if self.autosave_every_iters and (Iter % self.autosave_every_iters == 0):
                    self.save_checkpoint()

            # final autosave
            if self.autosave_every_iters:
                self.save_checkpoint()

            Puma_X = self.Best['X']
            Puma_C = self.Best['Cost']
            return {"gbest": Puma_X, "gbest_val": Puma_C, "best_per_iter": self.best_per_iter}
        except Exception as e:
            try:
                with open("po_exception.log", "a", encoding="utf-8") as f:
                    f.write(f"{time.time()} exception: {e}\n")
                    f.write(traceback.format_exc())
            except Exception:
                pass
            try:
                self.save_checkpoint(self.autosave_path.replace(".npz", "_onexception.npz"))
            except Exception:
                pass
            raise