"""
Dragonfly Optimization Algorithm (DOA) - translated from DOA.m (MATLAB)

API similar to other optimizers in this repo (CEO/PSO/PO):

Usage:
    doa = DOA(func, pop, dim, lb, ub, max_iter,
              autosave_every_iters=0, autosave_path="doa_checkpoint.npz", eval_delay=0.0, rng_seed=None)
    res = doa.optimize()
    res -> {"gbest": array, "gbest_val": float, "best_per_iter": list}

Notes:
- func(x) must accept 1D numpy array and return scalar objective
- This translation follows the structure of the provided DOA.m.
- Adds safe evaluations (exceptions -> +inf), optional eval_delay and autosave/checkpointing.
"""
import numpy as np
import time
import os
import traceback

class DOA:
    def __init__(self, func, pop, dim, lb, ub, max_iter,
                 autosave_every_iters=0, autosave_path="doa_checkpoint.npz", eval_delay=0.0, rng_seed=None):
        self.func = func
        self.pop = int(pop)
        self.dim = int(dim)
        # support scalar or vector lb/ub
        self.lb = np.array(lb * np.ones(self.dim) if np.isscalar(lb) else lb, dtype=float)
        self.ub = np.array(ub * np.ones(self.dim) if np.isscalar(ub) else ub, dtype=float)
        self.max_iter = int(max_iter)

        self.autosave_every_iters = int(autosave_every_iters)
        self.autosave_path = autosave_path
        self.eval_delay = float(eval_delay)

        if rng_seed is not None:
            np.random.seed(rng_seed)

        # bookkeeping
        self.best_per_iter = []
        self.iter = 0

        # initialization
        self.x = np.random.uniform(self.lb, self.ub, (self.pop, self.dim))
        # evaluate initial fitness
        self._init_evaluate_population()
        # global best
        idx = int(np.argmin(self.fitness))
        self.sbest = self.x[idx].copy()
        self.fbest = float(self.fitness[idx])

    def _safe_eval(self, x):
        try:
            v = float(self.func(np.array(x, dtype=float)))
        except Exception as e:
            try:
                with open("failed_evals_doa.csv", "a", encoding="utf-8") as f:
                    f.write(f"{time.time()},{','.join(map(str, x))},exception,{str(e)}\n")
            except Exception:
                pass
            v = float("inf")
        if self.eval_delay:
            time.sleep(self.eval_delay)
        return v

    def _init_evaluate_population(self):
        self.fitness = np.full(self.pop, np.inf, dtype=float)
        for i in range(self.pop):
            self.fitness[i] = self._safe_eval(self.x[i])

    def save_checkpoint(self, path=None):
        if path is None:
            path = self.autosave_path
        try:
            np.savez(path,
                     x=self.x,
                     fitness=self.fitness,
                     sbest=self.sbest,
                     fbest=self.fbest,
                     best_per_iter=np.array(self.best_per_iter),
                     iter=self.iter,
                     lb=self.lb,
                     ub=self.ub)
        except Exception as e:
            try:
                with open("doa_save_error.log", "a", encoding="utf-8") as f:
                    f.write(f"{time.time()} save error: {e}\n")
                    f.write(traceback.format_exc())
            except Exception:
                pass

    def _boundary_handle(self, idx, dim_idx):
        """
        Boundary handling:
         - if dim > 15: replace with gene from a random other individual
         - else: random within bounds
        """
        if self.dim > 15:
            # choose a different individual
            sel = np.random.randint(0, self.pop - 1)
            if sel >= idx:
                sel += 1
            return self.x[sel, dim_idx]
        else:
            return np.random.rand() * (self.ub[dim_idx] - self.lb[dim_idx]) + self.lb[dim_idx]

    def optimize(self):
        try:
            pop = self.pop
            D = self.dim
            T = self.max_iter
            SELECT = np.arange(pop)

            # initialize group-best and other structures
            sbest = np.ones(D)
            fbest = float("inf")
            sbestd = np.ones((5, D))
            fbestd = np.full(5, float("inf"))
            fbest_history = np.full(T, np.inf, dtype=float)

            # set initial group bests by splitting existing population
            # (this will be updated during iterations)
            # Exploration phase: first 9*T/10 iterations
            n_explore = int(np.floor(9 * T / 10.0))
            if n_explore < 1:
                n_explore = 1

            for i_iter in range(n_explore):
                # divide into 5 groups, roughly equal
                for m in range(1, 6):  # m = 1..5
                    # calculate group indices (MATLAB used ranges based on fractions)
                    start = int(np.floor((m - 1) / 5.0 * pop))
                    end = int(np.floor(m / 5.0 * pop))  # exclusive end in Python
                    if end <= start:
                        end = min(start + 1, pop)
                    group_idx = np.arange(start, end)

                    # compute k: random integer in [ceil(D/(8*m)), ceil(D/(3*m))]
                    low = int(np.ceil(D / (8.0 * m)))
                    high = int(np.ceil(D / (3.0 * m)))
                    low = max(1, low)
                    high = max(low, high)

                    # find best in group (update sbestd and fbestd)
                    for j in group_idx:
                        val = self._safe_eval(self.x[j])
                        # update stored fitness as in original (it re-evaluated)
                        # keep per-group best
                        if val < fbestd[m - 1]:
                            sbestd[m - 1, :] = self.x[j].copy()
                            fbestd[m - 1] = val

                    # Memory strategy and perturbation for each member in group
                    for j in group_idx:
                        # memory
                        self.x[j, :] = sbestd[m - 1, :].copy()
                        # randomly choose k dims
                        k = np.random.randint(low, high + 1) if high >= low else low
                        in_idx = np.random.permutation(D)[:k]
                        if np.random.rand() < 0.9:
                            for h in in_idx:
                                perturb = (np.random.rand() * (self.ub[h] - self.lb[h]) + self.lb[h]) * ((np.cos((1 * (i_iter + 1) + T / 10.0) * np.pi / T) + 1.0) / 2.0)
                                self.x[j, h] = self.x[j, h] + perturb
                                if (self.x[j, h] > self.ub[h]) or (self.x[j, h] < self.lb[h]):
                                    self.x[j, h] = self._boundary_handle(j, h)
                        else:
                            # copy gene from random individual
                            for h in in_idx:
                                sel = np.random.randint(0, pop)
                                self.x[j, h] = self.x[sel, h]

                    # update global best if group best improved
                    if fbestd[m - 1] < fbest:
                        fbest = fbestd[m - 1]
                        sbest = sbestd[m - 1, :].copy()

                # After processing groups, record global fbest
                # also update population fitness array (optional)
                # We keep a quick update: recompute fitness only for changed individuals
                for p in range(pop):
                    self.fitness[p] = self._safe_eval(self.x[p])
                idxg = int(np.argmin(self.fitness))
                if self.fitness[idxg] < fbest:
                    fbest = float(self.fitness[idxg])
                    sbest = self.x[idxg].copy()

                fbest_history[i_iter] = fbest
                self.best_per_iter.append(fbest)
                self.iter = i_iter + 1
                if (i_iter + 1) % 10 == 0:
                    print(f"Iteration: {i_iter+1} Best Cost = {fbest:.6g}")
                if self.autosave_every_iters and ((i_iter + 1) % self.autosave_every_iters == 0):
                    self.save_checkpoint()

            # Exploitation phase: last T - n_explore iterations
            for i_iter in range(n_explore, T):
                # update global best by scanning population
                for p in range(pop):
                    valp = self._safe_eval(self.x[p])
                    if valp < fbest:
                        fbest = valp
                        sbest = self.x[p].copy()

                # for each individual apply exploitation operators
                fitness_local = np.full(pop, np.inf)
                for j in range(pop):
                    fitness_local[j] = self._safe_eval(self.x[j])
                    km = max(2, int(np.ceil(D / 3.0)))
                    k = np.random.randint(2, km + 1)
                    # memory = global best
                    self.x[j, :] = sbest.copy()
                    in_idx = np.random.permutation(D)[:k]
                    for h in in_idx:
                        perturb = (np.random.rand() * (self.ub[h] - self.lb[h]) + self.lb[h]) * ((np.cos((i_iter + 1) * np.pi / T) + 1.0) / 2.0)
                        self.x[j, h] = self.x[j, h] + perturb
                        if (self.x[j, h] > self.ub[h]) or (self.x[j, h] < self.lb[h]):
                            self.x[j, h] = self._boundary_handle(j, h)
                    # evaluate new candidate and accept if better
                    new_val = self._safe_eval(self.x[j])
                    if new_val < fitness_local[j]:
                        fitness_local[j] = new_val

                # update global best from fitness_local
                idxg = int(np.argmin(fitness_local))
                if fitness_local[idxg] < fbest:
                    fbest = float(fitness_local[idxg])
                    sbest = self.x[idxg].copy()

                fbest_history[i_iter] = fbest
                self.best_per_iter.append(fbest)
                self.iter = i_iter + 1
                if (i_iter + 1) % 10 == 0:
                    print(f"Iteration: {i_iter+1} Best Cost = {fbest:.6g}")
                if self.autosave_every_iters and ((i_iter + 1) % self.autosave_every_iters == 0):
                    self.save_checkpoint()

            # final save
            if self.autosave_every_iters:
                self.save_checkpoint()

            self.sbest = sbest.copy()
            self.fbest = float(fbest)
            # convert history to list
            hist_list = [float(x) for x in self.best_per_iter]
            return {"gbest": self.sbest, "gbest_val": self.fbest, "best_per_iter": hist_list}
        except Exception as e:
            try:
                with open("doa_exception.log", "a", encoding="utf-8") as f:
                    f.write(f"{time.time()} exception: {e}\n")
                    f.write(traceback.format_exc())
            except Exception:
                pass
            try:
                self.save_checkpoint(self.autosave_path.replace(".npz", "_onexception.npz"))
            except Exception:
                pass
            raise