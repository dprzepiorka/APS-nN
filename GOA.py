"""
Goat Optimization Algorithm (GOA) - Python implementation (CEO-compatible interface)

This implementation is adapted from the provided MATLAB GOA snippets and structured
to match the optimizer classes used elsewhere in the project (CEO-like API).

Usage:
    from GOA import GOA
    goa = GOA(func, n_goats, dim, lb, ub, max_iter,
              alpha=0.05, beta=0.5, jump_prob=0.1,
              autosave_every_iters=0, autosave_path="goa_checkpoint.npz",
              eval_delay=0.0, rng_seed=None, initial_solutions=None)
    res = goa.optimize()
    res -> {"gbest": array, "gbest_val": float, "best_per_iter": list}

Notes:
- func(x) must accept a 1D numpy array and return a scalar objective.
- lb/ub may be scalars or length-d iterables. They will be expanded to length dim arrays.
- Defaults follow the MATLAB snippet: alpha (exploration), beta (exploitation), J (jump prob).
- Implements parasite avoidance (replace weakest fraction) and optional random reinitialization.
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


class GOA:
    def __init__(
        self,
        func: Callable[[np.ndarray], float],
        n_goats,
        dim,
        lb,
        ub,
        max_iter,
        alpha: float = 0.05,
        beta: float = 0.5,
        jump_prob: float = 0.1,
        parasite_fraction: float = 0.2,
        autosave_every_iters: int = 0,
        autosave_path: str = "goa_checkpoint.npz",
        eval_delay: float = 0.0,
        rng_seed: Optional[int] = None,
        initial_solutions: Optional[np.ndarray] = None,
    ):
        """
        Initialize GOA optimizer.
        - func: objective function
        - n_goats: population size
        - dim: problem dimension
        - lb, ub: scalar or length-d bounds
        - max_iter: number of iterations
        - alpha: exploration coefficient (gaussian step scale)
        - beta: exploitation coefficient (move toward global best)
        - jump_prob: probability of jump towards a random goat (jump strategy)
        - parasite_fraction: fraction of weakest goats to reinitialize each iter
        """
        self.func = func
        self.n_goats = _to_int_scalar(n_goats, "n_goats")
        self.dim = _to_int_scalar(dim, "dim")
        self.max_iter = _to_int_scalar(max_iter, "max_iter")

        # bounds
        if np.isscalar(lb):
            self.lb = np.full(self.dim, float(lb))
        else:
            arr_lb = np.asarray(lb, dtype=float)
            if arr_lb.size == 1:
                self.lb = np.full(self.dim, float(arr_lb.item()))
            elif arr_lb.size == self.dim:
                self.lb = arr_lb.copy()
            else:
                raise ValueError(f"lb must be scalar or length-{self.dim} array")

        if np.isscalar(ub):
            self.ub = np.full(self.dim, float(ub))
        else:
            arr_ub = np.asarray(ub, dtype=float)
            if arr_ub.size == 1:
                self.ub = np.full(self.dim, float(arr_ub.item()))
            elif arr_ub.size == self.dim:
                self.ub = arr_ub.copy()
            else:
                raise ValueError(f"ub must be scalar or length-{self.dim} array")

        if np.any(self.ub <= self.lb):
            raise ValueError("Each ub must be greater than lb for all dimensions")

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.jump_prob = float(jump_prob)
        self.parasite_fraction = float(parasite_fraction)

        self.autosave_every_iters = int(autosave_every_iters)
        self.autosave_path = autosave_path if autosave_every_iters else None
        self.eval_delay = float(eval_delay)

        if rng_seed is not None:
            np.random.seed(int(rng_seed))

        # population: if initial_solutions given, validate shape
        if initial_solutions is not None:
            arr = np.asarray(initial_solutions, dtype=float)
            if arr.shape != (self.n_goats, self.dim):
                raise ValueError("initial_solutions must have shape (n_goats, dim)")
            self.pop = arr.copy()
        else:
            self.pop = np.random.rand(self.n_goats, self.dim) * (self.ub - self.lb) + self.lb

        # fitness and bookkeeping
        self.fitness = np.full(self.n_goats, np.inf, dtype=float)
        self.best_per_iter: list[float] = []
        self.iter = 0

        # evaluate initial population
        for i in range(self.n_goats):
            self.fitness[i] = self._safe_eval(self.pop[i, :])
        best_idx = int(np.argmin(self.fitness))
        self.gbest = self.pop[best_idx, :].copy()
        self.gbest_val = float(self.fitness[best_idx])
        self.best_per_iter.append(self.gbest_val)

    def _safe_eval(self, x: np.ndarray) -> float:
        try:
            v = float(self.func(np.asarray(x, dtype=float)))
        except Exception as e:
            try:
                with open("failed_evals_goa.csv", "a", encoding="utf-8") as f:
                    f.write(f"{time.time()},{','.join(map(str, x.tolist()))},exception,{str(e)}\n")
            except Exception:
                pass
            v = float("inf")
        if self.eval_delay:
            time.sleep(self.eval_delay)
        return v

    def _clip(self, arr: np.ndarray) -> np.ndarray:
        return np.minimum(np.maximum(arr, self.lb), self.ub)

    def save_checkpoint(self, path: Optional[str] = None) -> None:
        path = path or self.autosave_path
        if not path:
            return
        try:
            np.savez(
                path,
                pop=self.pop,
                fitness=self.fitness,
                gbest=self.gbest,
                gbest_val=self.gbest_val,
                iter=self.iter,
                best_per_iter=np.array(self.best_per_iter),
                lb=self.lb,
                ub=self.ub,
            )
        except Exception as e:
            try:
                with open("goa_save_error.log", "a", encoding="utf-8") as f:
                    f.write(f"{time.time()} save error: {e}\n")
                    f.write(traceback.format_exc())
            except Exception:
                pass

    def optimize(self) -> dict:
        """
        Run GOA optimization loop. Returns {"gbest","gbest_val","best_per_iter"}.
        """
        try:
            N = self.n_goats
            dim = self.dim

            for it in range(1, self.max_iter + 1):
                self.iter = it

                # iteration-specific small adjustments (optional annealing of alpha/beta)
                # allow slight decay to encourage exploitation later
                alpha_t = self.alpha * (1.0 - (it / float(self.max_iter)) * 0.5)
                beta_t = self.beta * (0.5 + (it / float(self.max_iter)) * 0.5)

                for i in range(N):
                    # Exploration: gaussian perturbation scaled by domain range
                    if np.random.rand() < 0.5:
                        step = np.random.randn(dim) * alpha_t * (self.ub - self.lb)
                        self.pop[i, :] = self.pop[i, :] + step

                    # Exploitation: move toward global best
                    if np.random.rand() >= 0.5:
                        self.pop[i, :] = self.pop[i, :] + beta_t * (self.gbest - self.pop[i, :])

                    # Jump strategy: occasionally move toward a random peer
                    if np.random.rand() < self.jump_prob:
                        rand_idx = np.random.randint(0, N)
                        self.pop[i, :] = self.pop[i, :] + self.jump_prob * (self.pop[rand_idx, :] - self.pop[i, :])

                    # enforce bounds
                    self.pop[i, :] = self._clip(self.pop[i, :])

                # evaluate fitness after moves
                for i in range(N):
                    self.fitness[i] = self._safe_eval(self.pop[i, :])

                # Parasite avoidance: replace weakest fraction with random solutions
                k_weak = int(np.floor(self.parasite_fraction * N))
                if k_weak > 0:
                    weakest_idx = np.argsort(self.fitness)[-k_weak:]
                    # reinitialize weakest
                    self.pop[weakest_idx, :] = np.random.rand(k_weak, dim) * (self.ub - self.lb) + self.lb
                    # evaluate reinitialized goats
                    for idx in weakest_idx:
                        self.fitness[idx] = self._safe_eval(self.pop[int(idx), :])

                # Update global best
                cur_best_idx = int(np.argmin(self.fitness))
                cur_best_val = float(self.fitness[cur_best_idx])
                if cur_best_val < self.gbest_val:
                    self.gbest_val = cur_best_val
                    self.gbest = self.pop[cur_best_idx, :].copy()

                self.best_per_iter.append(self.gbest_val)

                # print progress occasionally
                if it % 10 == 0 or it == 1 or it == self.max_iter:
                    print(f"GOA Iter {it}/{self.max_iter} Best = {self.gbest_val:.6g}")

                # autosave
                if self.autosave_every_iters and (it % self.autosave_every_iters == 0):
                    self.save_checkpoint()

            # final autosave
            if self.autosave_every_iters:
                self.save_checkpoint()

            return {"gbest": self.gbest.copy(), "gbest_val": float(self.gbest_val), "best_per_iter": list(self.best_per_iter)}
        except Exception as e:
            try:
                with open("goa_exception.log", "a", encoding="utf-8") as f:
                    f.write(f"{time.time()} exception: {e}\n")
                    f.write(traceback.format_exc())
            except Exception:
                pass
            if self.autosave_every_iters:
                try:
                    self.save_checkpoint((self.autosave_path or "goa_checkpoint.npz").replace(".npz", "_onexception.npz"))
                except Exception:
                    pass
            raise


__all__ = ["GOA"]