"""
Birds of Prey-Based Optimization (BPBO) - safe Python module

Provides class BPBO with interface:
    bpbo = BPBO(func, n_pop, dim, lb, ub, max_iter, Pi=0.7, autosave_every_iters=0, autosave_path="bpbo_checkpoint.npz", eval_delay=0.0, rng_seed=None)
    res = bpbo.optimize()  # -> {"gbest": array, "gbest_val": float, "best_per_iter": list}

No heavy top-level execution — safe to import.
"""
import numpy as np
import time
import traceback
import os

class BPB:
    def __init__(self, func, n_pop, dim, lb, ub, max_iter,
                 Pi=0.7, autosave_every_iters=0, autosave_path="bpbo_checkpoint.npz",
                 eval_delay=0.0, rng_seed=None):
        self.func = func
        self.n_pop = int(n_pop)
        self.dim = int(dim)
        # allow scalar lb/ub or vector
        if np.isscalar(lb):
            self.lb = np.full(self.dim, float(lb))
        else:
            self.lb = np.array(lb, dtype=float)
        if np.isscalar(ub):
            self.ub = np.full(self.dim, float(ub))
        else:
            self.ub = np.array(ub, dtype=float)

        self.max_iter = int(max_iter)
        self.Pi = float(Pi)

        self.autosave_every_iters = int(autosave_every_iters)
        self.autosave_path = autosave_every_iters and autosave_path or None
        self.eval_delay = float(eval_delay)

        if rng_seed is not None:
            np.random.seed(rng_seed)

        # population container: list of dicts {'X':..., 'Cost':...}
        self.Pop = []
        self.best_per_iter = []
        self.iter = 0

        # initialize population (deferred to init_population so import is safe)
        self._init_population()

    def _safe_eval(self, x):
        try:
            v = float(self.func(np.array(x, dtype=float)))
        except Exception as e:
            try:
                with open("failed_evals_bpbo.csv", "a", encoding="utf-8") as f:
                    f.write(f"{time.time()},{','.join(map(str, x))},exception,{str(e)}\n")
            except Exception:
                pass
            v = float("inf")
        if self.eval_delay:
            time.sleep(self.eval_delay)
        return v

    def _init_population(self):
        # initialize population only when instance created
        self.Pop = []
        for i in range(self.n_pop):
            x = np.random.uniform(self.lb, self.ub, self.dim)
            c = self._safe_eval(x)
            self.Pop.append({'X': x, 'Cost': c})
        # find initial Prey (best)
        costs = [p['Cost'] for p in self.Pop]
        idx = int(np.argmin(costs))
        self.Prey = self.Pop[idx].copy()

    def save_checkpoint(self, path=None):
        path = path or self.autosave_path
        if not path:
            return
        try:
            np.savez(path,
                     Pop_X=np.array([p['X'] for p in self.Pop]),
                     Pop_C=np.array([p['Cost'] for p in self.Pop]),
                     Prey_X=self.Prey['X'],
                     Prey_C=self.Prey['Cost'],
                     iter=self.iter,
                     best_per_iter=np.array(self.best_per_iter))
        except Exception as e:
            try:
                with open("bpbo_save_error.log", "a", encoding="utf-8") as f:
                    f.write(f"{time.time()} save error: {e}\n")
                    f.write(traceback.format_exc())
            except Exception:
                pass

    def _clip(self, x):
        return np.minimum(np.maximum(x, self.lb), self.ub)

    def optimize(self):
        try:
            MaxIt = self.max_iter
            VarMin = self.lb
            VarMax = self.ub
            nPop = self.n_pop

            for it in range(1, MaxIt + 1):
                self.iter = it
                # Compute Mean of population positions
                Mean = np.mean(np.vstack([p['X'] for p in self.Pop]), axis=0)

                # ensure Prey is current best
                costs = [p['Cost'] for p in self.Pop]
                idx_best = int(np.argmin(costs))
                self.Prey = self.Pop[idx_best].copy()

                # iterate over population to create new solutions
                for i in range(nPop):
                    cur = self.Pop[i]
                    if np.random.rand() < self.Pi:
                        # nested random choices as in original
                        if np.random.rand() < np.random.rand():
                            M00 = int(round(1 + np.random.rand()))
                            newX = cur['X'] + np.random.rand(self.dim) * (self.Prey['X'] - M00 * cur['X'])
                        elif np.random.rand() < np.random.rand():
                            M01 = int(round(1 + np.random.rand()))
                            newX = Mean + np.random.rand(self.dim) * (self.Prey['X'] - M01 * Mean)
                        else:
                            M02 = int(round(1 + np.random.rand()))
                            last = self.Pop[-1]['X']
                            newX = cur['X'] + np.random.rand(self.dim) * (cur['X'] - M02 * last)
                    else:
                        newX = cur['X'] + np.random.rand() * np.random.uniform(VarMin, VarMax, self.dim)

                    newX = self._clip(newX)
                    newC = self._safe_eval(newX)

                    if newC < cur['Cost']:
                        self.Pop[i] = {'X': newX, 'Cost': newC}
                        if newC < self.Prey['Cost']:
                            self.Prey = self.Pop[i].copy()

                # sort and record
                self.Pop.sort(key=lambda p: p['Cost'])
                best_cost = self.Pop[0]['Cost']
                self.best_per_iter.append(best_cost)

                if self.autosave_every_iters and (it % self.autosave_every_iters == 0):
                    self.save_checkpoint()

                if it % 10 == 0 or it == 1 or it == MaxIt:
                    print(f"Iteration {it}: Best Cost = {best_cost:.6g}")

            gbest = self.Pop[0]['X'].copy()
            gbest_val = float(self.Pop[0]['Cost'])
            return {"gbest": gbest, "gbest_val": gbest_val, "best_per_iter": list(self.best_per_iter)}
        except Exception as e:
            try:
                with open("bpbo_exception.log", "a", encoding="utf-8") as f:
                    f.write(f"{time.time()} exception: {e}\n")
                    f.write(traceback.format_exc())
            except Exception:
                pass
            try:
                self.save_checkpoint((self.autosave_path or "bpbo_checkpoint.npz").replace(".npz", "_onexception.npz"))
            except Exception:
                pass
            raise

# optional: expose only BPBO in from BPBO import *
__all__ = ["BPBO"]