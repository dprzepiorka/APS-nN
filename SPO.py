"""
Stochastic Paint Optimizer (SPO) - Python translation of SPO.m

Interface consistent with other optimizers in this repo (CEO/PSO/PO/DOA):

Usage:
    spo = SPO(func, n_colors, dim, lb, ub, max_iter,
              autosave_every_iters=0, autosave_path="spo_checkpoint.npz", eval_delay=0.0, rng_seed=None)
    res = spo.optimize()
    res -> {"gbest": array, "gbest_val": float, "best_per_iter": list}

Notes:
- func(x) must accept a 1D numpy array and return scalar objective.
- lb/ub may be scalars or length-D iterables. They will be expanded to length D arrays.
- This implementation follows the structure of the provided SPO.m and adds:
  - safe evaluations (exceptions -> +inf),
  - optional eval_delay,
  - autosave / checkpointing.
"""
import numpy as np
import time
import os
import traceback

class SPO:
    def __init__(self, func, n_colors, dim, lb, ub, max_iter,
                 autosave_every_iters=0, autosave_path="spo_checkpoint.npz", eval_delay=0.0, rng_seed=None):
        self.func = func
        self.n_colors = int(n_colors)
        self.dim = int(dim)
        # Support scalar or vector lb/ub
        if np.isscalar(lb):
            self.lb = np.full(self.dim, float(lb))
        else:
            self.lb = np.array(lb, dtype=float)
        if np.isscalar(ub):
            self.ub = np.full(self.dim, float(ub))
        else:
            self.ub = np.array(ub, dtype=float)
        self.max_iter = int(max_iter)

        self.autosave_every_iters = int(autosave_every_iters)
        self.autosave_path = autosave_path
        self.eval_delay = float(eval_delay)

        if rng_seed is not None:
            np.random.seed(rng_seed)

        # initialize population (Colors) and evaluate
        self.Colors = np.random.uniform(self.lb, self.ub, (self.n_colors, self.dim))
        self.Fun_eval = np.array([self._safe_eval(self.Colors[i]) for i in range(self.n_colors)], dtype=float)

        # bookkeeping
        idx = int(np.argmin(self.Fun_eval))
        self.BestColors = self.Colors[idx].copy()
        self.BestFitness = float(self.Fun_eval[idx])
        self.best_per_iter = []
        self.iter = 0

    def _safe_eval(self, x):
        try:
            v = float(self.func(np.array(x, dtype=float)))
        except Exception as e:
            try:
                with open("failed_evals_spo.csv", "a", encoding="utf-8") as f:
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
                     Colors=self.Colors,
                     Fun_eval=self.Fun_eval,
                     BestColors=self.BestColors,
                     BestFitness=self.BestFitness,
                     iter=self.iter,
                     best_per_iter=np.array(self.best_per_iter))
        except Exception as e:
            try:
                with open("spo_save_error.log", "a", encoding="utf-8") as f:
                    f.write(f"{time.time()} save error: {e}\n")
                    f.write(traceback.format_exc())
            except Exception:
                pass

    def _bound(self, x):
        x = np.minimum(x, self.ub)
        x = np.maximum(x, self.lb)
        return x

    def optimize(self):
        try:
            Colors_Number = self.n_colors
            Var_Number = self.dim
            MaxIter = self.max_iter

            # compute group sizes
            N1stColors = Colors_Number // 3
            N2ndColors = Colors_Number // 3
            N3rdColors = Colors_Number - N1stColors - N2ndColors
            if N1stColors < 1:
                N1stColors = 1
            if N2ndColors < 1 and Colors_Number > 1:
                N2ndColors = 1
            if N3rdColors < 1:
                N3rdColors = max(1, Colors_Number - N1stColors - N2ndColors)

            for Iter in range(1, MaxIter + 1):
                # sort by fitness ascending
                order = np.argsort(self.Fun_eval)
                self.Colors = self.Colors[order]
                self.Fun_eval = self.Fun_eval[order]

                Group1st = self.Colors[0:N1stColors]
                Group2nd = self.Colors[N1stColors:N1stColors + N2ndColors]
                Group3rd = self.Colors[N1stColors + N2ndColors:Colors_Number]

                NewColors_list = []
                NewFuns_list = []

                # for each color generate new candidates (4 combinations in MATLAB)
                for ind in range(Colors_Number):
                    # Complement Combination
                    Id1 = np.random.randint(0, N1stColors)
                    Id2 = np.random.randint(0, N3rdColors)
                    new1 = self.Colors[ind] + np.random.rand(Var_Number) * (Group1st[Id1] - Group3rd[Id2])
                    new1 = self._bound(new1)
                    new1_cost = self._safe_eval(new1)
                    NewColors_list.append(new1); NewFuns_list.append(new1_cost)

                    # Analog Combination
                    if ind < N1stColors:
                        ids = np.random.randint(0, N1stColors, size=2)
                        AnalogGroup = Group1st
                    elif ind < N1stColors + N2ndColors:
                        ids = np.random.randint(0, N2ndColors, size=2)
                        AnalogGroup = Group2nd
                    else:
                        ids = np.random.randint(0, N3rdColors, size=2)
                        AnalogGroup = Group3rd
                    new2 = self.Colors[ind] + np.random.rand(Var_Number) * (AnalogGroup[ids[1]] - AnalogGroup[ids[0]])
                    new2 = self._bound(new2)
                    new2_cost = self._safe_eval(new2)
                    NewColors_list.append(new2); NewFuns_list.append(new2_cost)

                    # Triangle Combination
                    Id1 = np.random.randint(0, N1stColors)
                    Id2 = np.random.randint(0, N2ndColors)
                    Id3 = np.random.randint(0, N3rdColors)
                    tri_sum = Group1st[Id1] + Group2nd[Id2] + Group3rd[Id3]
                    new3 = self.Colors[ind] + np.random.rand(Var_Number) * (tri_sum / 3.0)
                    new3 = self._bound(new3)
                    new3_cost = self._safe_eval(new3)
                    NewColors_list.append(new3); NewFuns_list.append(new3_cost)

                    # Rectangle Combination
                    Id1 = np.random.randint(0, N1stColors)
                    Id2 = np.random.randint(0, N2ndColors)
                    Id3 = np.random.randint(0, N3rdColors)
                    Id4 = np.random.randint(0, Colors_Number)
                    rect = (np.random.rand(Var_Number) * Group1st[Id1] +
                            np.random.rand(Var_Number) * Group2nd[Id2] +
                            np.random.rand(Var_Number) * Group3rd[Id3] +
                            np.random.rand(Var_Number) * self.Colors[Id4]) / 4.0
                    new4 = self.Colors[ind] + rect
                    new4 = self._bound(new4)
                    new4_cost = self._safe_eval(new4)
                    NewColors_list.append(new4); NewFuns_list.append(new4_cost)

                # merge populations and pick top Colors_Number
                if NewColors_list:
                    new_arr = np.vstack(NewColors_list)
                    new_costs = np.array(NewFuns_list, dtype=float)
                    self.Colors = np.vstack([self.Colors, new_arr])
                    self.Fun_eval = np.concatenate([self.Fun_eval, new_costs])

                # sort and keep the best Colors_Number
                order = np.argsort(self.Fun_eval)
                self.Colors = self.Colors[order]
                self.Fun_eval = self.Fun_eval[order]
                # get best
                best_idx = 0
                SortedFit = float(self.Fun_eval[best_idx])
                self.BestColors = self.Colors[best_idx].copy()
                self.BestFitness = SortedFit
                # trim
                self.Colors = self.Colors[:Colors_Number]
                self.Fun_eval = self.Fun_eval[:Colors_Number]

                # record history
                self.best_per_iter.append(SortedFit)
                self.iter = Iter

                # print and optionally save
                print(f" Iter= {Iter}  BestCost= {SortedFit}")
                if self.autosave_every_iters and (Iter % self.autosave_every_iters == 0):
                    self.save_checkpoint()

            # final save
            if self.autosave_every_iters:
                self.save_checkpoint()

            return {"gbest": self.BestColors, "gbest_val": self.BestFitness, "best_per_iter": list(self.best_per_iter)}

        except Exception as e:
            try:
                with open("spo_exception.log", "a", encoding="utf-8") as f:
                    f.write(f"{time.time()} exception: {e}\n")
                    f.write(traceback.format_exc())
            except Exception:
                pass
            try:
                self.save_checkpoint(self.autosave_path.replace(".npz", "_onexception.npz"))
            except Exception:
                pass
            raise