"""
Celestial Orbit Optimization (COO) - Python implementation

This implements a COO optimizer based on the provided MATLAB
COO_MultiObjective_Optimizer.m. The Python version supports both
single-objective and multi-objective problems:

- If the objective returns a scalar, COO runs as a single-objective
  optimizer and returns the best solution and value.
- If the objective returns a vector/array (length > 1) COO treats the
  problem as multi-objective and returns the Pareto front.

Usage (single-objective):
    coo = COO(func, n_pop, dim, lb, ub, max_iter, rng_seed=0)
    res = coo.optimize()
    res -> {"gbest": array, "gbest_val": float, "best_per_iter": list}

Usage (multi-objective):
    coo = COO(func, n_pop, dim, lb, ub, max_iter)
    res = coo.optimize()
    res -> {"pareto": ndarray (k x dim), "pareto_fitness": ndarray (k x m),
            "pareto_history": list of ints (pareto front sizes per iter)}

Notes:
- func(x) must accept 1D numpy array and return scalar (single-objective)
  or iterable/1D-array (multi-objective).
- lb/ub may be scalars or arrays of length dim.
- Safe objective evaluation: exceptions are logged and result treated as +inf.
- No top-level execution so file is safe to import.
"""
from __future__ import annotations
import numpy as np
import time
import traceback
from typing import Callable, Optional


def _to_int_scalar(x, name="value"):
    if np.isscalar(x):
        return int(x)
    arr = np.asarray(x)
    if arr.size == 1:
        return int(arr.item())
    raise TypeError(f"'{name}' must be a scalar or 1-element array/list, got shape {arr.shape}")


class COO:
    def __init__(
        self,
        func: Callable[[np.ndarray], object],
        n_pop,
        dim,
        lb,
        ub,
        max_iter,
        autosave_every_iters: int = 0,
        autosave_path: str = "coo_checkpoint.npz",
        eval_delay: float = 0.0,
        rng_seed: Optional[int] = None,
    ):
        """
        Initialize COO optimizer.

        Parameters:
        - func: objective function f(x). Returns scalar or vector-like.
        - n_pop: population size
        - dim: problem dimension
        - lb, ub: scalar or length-d bounds
        - max_iter: number of iterations
        - autosave_every_iters, autosave_path: optional checkpointing
        - eval_delay: seconds to sleep after each objective evaluation (for slow functions)
        - rng_seed: optional seed
        """
        self.func = func
        self.n_pop = _to_int_scalar(n_pop, "n_pop")
        self.dim = _to_int_scalar(dim, "dim")
        self.max_iter = _to_int_scalar(max_iter, "max_iter")

        # bounds handling
        if np.isscalar(lb):
            self.lb = np.full(self.dim, float(lb))
        else:
            a = np.asarray(lb, dtype=float)
            if a.size == 1:
                self.lb = np.full(self.dim, float(a.item()))
            elif a.size == self.dim:
                self.lb = a.copy()
            else:
                raise ValueError("lb must be scalar or length-d array")

        if np.isscalar(ub):
            self.ub = np.full(self.dim, float(ub))
        else:
            a = np.asarray(ub, dtype=float)
            if a.size == 1:
                self.ub = np.full(self.dim, float(a.item()))
            elif a.size == self.dim:
                self.ub = a.copy()
            else:
                raise ValueError("ub must be scalar or length-d array")

        if np.any(self.ub <= self.lb):
            raise ValueError("Each ub must be greater than lb for all dimensions")

        self.autosave_every_iters = int(autosave_every_iters)
        self.autosave_path = autosave_path if autosave_every_iters else None
        self.eval_delay = float(eval_delay)

        if rng_seed is not None:
            np.random.seed(int(rng_seed))

        # initialize population and velocities
        self.pop = np.random.rand(self.n_pop, self.dim) * (self.ub - self.lb) + self.lb
        self.vel = np.zeros((self.n_pop, self.dim), dtype=float)

        # history and bookkeeping
        self.iter = 0
        self.best_per_iter = []

        # evaluate first individual to detect multi-objective or single-objective
        v0 = self._safe_eval(self.pop[0, :])
        self.is_multi = False
        try:
            arr0 = np.asarray(v0)
            if arr0.ndim >= 1 and arr0.size > 1:
                self.is_multi = True
                self.obj_dim = arr0.size
            else:
                self.is_multi = False
        except Exception:
            self.is_multi = False

        # fitness container
        if self.is_multi:
            self.fitness = np.zeros((self.n_pop, self.obj_dim), dtype=float)
        else:
            self.fitness = np.full(self.n_pop, np.inf, dtype=float)

        # evaluate whole initial population
        for i in range(self.n_pop):
            val = self._safe_eval(self.pop[i, :])
            if self.is_multi:
                self.fitness[i, :] = np.asarray(val, dtype=float)
            else:
                self.fitness[i] = float(val)

        # initial best or pareto
        if self.is_multi:
            fronts = self._nondominated_sort(self.fitness)
            self.best_front = fronts[0] if len(fronts) > 0 else list(range(self.n_pop))
            self.pareto_pop = self.pop[self.best_front, :].copy()
            self.pareto_fit = self.fitness[self.best_front, :].copy()
            self.best_per_iter = [len(self.best_front)]
        else:
            idx = int(np.argmin(self.fitness))
            self.gbest = self.pop[idx, :].copy()
            self.gbest_val = float(self.fitness[idx])
            self.best_per_iter = [self.gbest_val]

    def _safe_eval(self, x: np.ndarray):
        try:
            res = self.func(np.asarray(x, dtype=float))
        except Exception as e:
            try:
                with open("failed_evals_coo.csv", "a", encoding="utf-8") as f:
                    f.write(f"{time.time()},{','.join(map(str, x.tolist()))},exception,{str(e)}\n")
            except Exception:
                pass
            # return +inf for scalar or large positive vector for multiobj
            if getattr(self, "is_multi", False):
                return np.full(getattr(self, "obj_dim", 1), np.inf)
            else:
                return float("inf")
        if self.eval_delay:
            time.sleep(self.eval_delay)
        return res

    @staticmethod
    def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
        # a dominates b if a <= b in all objectives and < in at least one
        return np.all(a <= b) and np.any(a < b)

    def _nondominated_sort(self, fits: np.ndarray):
        """
        Fast non-dominated sorting (simple O(n^2) implementation).
        Returns list of fronts, each front is a list of indices.
        """
        n = fits.shape[0]
        S = [list() for _ in range(n)]
        n_dom = np.zeros(n, dtype=int)
        fronts = []

        for p in range(n):
            for q in range(n):
                if p == q:
                    continue
                if self._dominates(fits[p], fits[q]):
                    S[p].append(q)
                elif self._dominates(fits[q], fits[p]):
                    n_dom[p] += 1
            if n_dom[p] == 0:
                if len(fronts) == 0:
                    fronts.append([p])
                else:
                    fronts[0].append(p)

        i = 0
        while i < len(fronts):
            next_front = []
            for p in fronts[i]:
                for q in S[p]:
                    n_dom[q] -= 1
                    if n_dom[q] == 0:
                        next_front.append(q)
            if len(next_front) == 0:
                break
            fronts.append(next_front)
            i += 1
        return fronts

    def _clip(self, arr: np.ndarray) -> np.ndarray:
        return np.minimum(np.maximum(arr, self.lb), self.ub)

    def save_checkpoint(self, path: Optional[str] = None) -> None:
        path = path or self.autosave_path
        if not path:
            return
        try:
            if self.is_multi:
                np.savez(
                    path,
                    pop=self.pop,
                    vel=self.vel,
                    pareto_pop=self.pareto_pop,
                    pareto_fit=self.pareto_fit,
                    iter=self.iter,
                    pareto_history=np.array(self.best_per_iter),
                    lb=self.lb,
                    ub=self.ub,
                )
            else:
                np.savez(
                    path,
                    pop=self.pop,
                    vel=self.vel,
                    gbest=self.gbest,
                    gbest_val=self.gbest_val,
                    iter=self.iter,
                    best_per_iter=np.array(self.best_per_iter),
                    lb=self.lb,
                    ub=self.ub,
                )
        except Exception as e:
            try:
                with open("coo_save_error.log", "a", encoding="utf-8") as f:
                    f.write(f"{time.time()} save error: {e}\n")
                    f.write(traceback.format_exc())
            except Exception:
                pass

    def optimize(self):
        """
        Run COO optimization loop.
        Returns:
         - single-objective: {"gbest", "gbest_val", "best_per_iter"}
         - multi-objective: {"pareto", "pareto_fitness", "pareto_history"}
        """
        try:
            for it in range(1, self.max_iter + 1):
                self.iter = it
                G = 1.0 - float(it) / float(self.max_iter)
                if self.is_multi:
                    # recompute best front and center of best front
                    fronts = self._nondominated_sort(self.fitness)
                    best_front = fronts[0] if len(fronts) > 0 else list(range(self.n_pop))
                    center = np.mean(self.pop[best_front, :], axis=0)
                else:
                    center = np.mean(self.pop, axis=0)

                # update velocities and positions
                for i in range(self.n_pop):
                    r = np.random.rand(self.dim)
                    accel = G * (center - self.pop[i, :])
                    self.vel[i, :] = 0.5 * self.vel[i, :] + accel + 0.01 * r * (self.ub - self.lb)
                    self.pop[i, :] = self.pop[i, :] + self.vel[i, :]

                    # apply bounds
                    self.pop[i, :] = self._clip(self.pop[i, :])

                    # evaluate
                    val = self._safe_eval(self.pop[i, :])
                    if self.is_multi:
                        self.fitness[i, :] = np.asarray(val, dtype=float)
                    else:
                        self.fitness[i] = float(val)

                # update best/pareto
                if self.is_multi:
                    fronts = self._nondominated_sort(self.fitness)
                    best_front = fronts[0] if len(fronts) > 0 else list(range(self.n_pop))
                    self.pareto_pop = self.pop[best_front, :].copy()
                    self.pareto_fit = self.fitness[best_front, :].copy()
                    self.best_per_iter.append(len(best_front))
                else:
                    idx = int(np.argmin(self.fitness))
                    if self.fitness[idx] < self.gbest_val:
                        self.gbest_val = float(self.fitness[idx])
                        self.gbest = self.pop[idx, :].copy()
                    self.best_per_iter.append(self.gbest_val)

                # optional autosave
                if self.autosave_every_iters and (it % self.autosave_every_iters == 0):
                    self.save_checkpoint()

                # progress print
                if it % 10 == 0 or it == 1 or it == self.max_iter:
                    if self.is_multi:
                        print(f"COO Iter {it}/{self.max_iter} Pareto size: {len(self.pareto_pop)}")
                    else:
                        print(f"COO Iter {it}/{self.max_iter} Best = {self.gbest_val:.6g}")

            # prepare output
            if self.is_multi:
                return {"pareto": self.pareto_pop.copy(), "pareto_fitness": self.pareto_fit.copy(), "pareto_history": list(self.best_per_iter)}
            else:
                return {"gbest": self.gbest.copy(), "gbest_val": float(self.gbest_val), "best_per_iter": list(self.best_per_iter)}
        except Exception as e:
            try:
                with open("coo_exception.log", "a", encoding="utf-8") as f:
                    f.write(f"{time.time()} exception: {e}\n")
                    f.write(traceback.format_exc())
            except Exception:
                pass
            if self.autosave_every_iters:
                try:
                    self.save_checkpoint((self.autosave_path or "coo_checkpoint.npz").replace(".npz", "_onexception.npz"))
                except Exception:
                    pass
            raise


__all__ = ["COO"]