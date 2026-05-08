"""
Owl Search Algorithm (OSA) - Python translation of OSA.m

Interface:
    from OSA import OSA
    osa = OSA(func, n_pop, dim, lb, ub, max_iter,
              autosave_every_iters=0, autosave_path="osa_checkpoint.npz",
              eval_delay=0.0, rng_seed=None)
    res = osa.optimize()
    res -> {"gbest": array, "gbest_val": float, "best_per_iter": list}

Notes:
- func(x) should accept a 1D numpy array and return a scalar objective.
- lb/ub may be scalars or length-d iterables. They will be expanded to length dim arrays.
- The implementation follows the MATLAB OSA.m logic but adds safe evaluation, optional eval_delay and autosave/checkpointing.
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
    raise TypeError(f"'{name}' must be a scalar or 1-element array/list, got shape {arr.shape} and value {x!r}")

class OSA:
    def __init__(
        self,
        func: Callable[[np.ndarray], float],
        n_pop,
        dim,
        lb,
        ub,
        max_iter,
        autosave_every_iters: int = 0,
        autosave_path: str = "osa_checkpoint.npz",
        eval_delay: float = 0.0,
        rng_seed: Optional[int] = None,
    ):
        self.func = func
        self.n_pop = _to_int_scalar(n_pop, "n_pop")
        self.dim = _to_int_scalar(dim, "dim")
        self.max_iter = _to_int_scalar(max_iter, "max_iter")

        # bounds normalization
        if np.isscalar(lb):
            self.lb = np.full(self.dim, float(lb))
        else:
            arr_lb = np.asarray(lb, dtype=float)
            if arr_lb.size == 1:
                self.lb = np.full(self.dim, float(arr_lb.item()))
            elif arr_lb.size == self.dim:
                self.lb = arr_lb.copy()
            else:
                raise ValueError(f"lb must be scalar or length-{self.dim} array, got {arr_lb.shape}")

        if np.isscalar(ub):
            self.ub = np.full(self.dim, float(ub))
        else:
            arr_ub = np.asarray(ub, dtype=float)
            if arr_ub.size == 1:
                self.ub = np.full(self.dim, float(arr_ub.item()))
            elif arr_ub.size == self.dim:
                self.ub = arr_ub.copy()
            else:
                raise ValueError(f"ub must be scalar or length-{self.dim} array, got {arr_ub.shape}")

        if np.any(self.ub <= self.lb):
            raise ValueError("Each ub must be greater than lb for all dimensions")

        self.autosave_every_iters = int(autosave_every_iters)
        self.autosave_path = autosave_path if autosave_every_iters else None
        self.eval_delay = float(eval_delay)

        if rng_seed is not None:
            np.random.seed(int(rng_seed))

        # population: lists of dicts like MATLAB struct
        self.owl_pos = np.zeros((self.n_pop, self.dim), dtype=float)
        self.owl_cost = np.full(self.n_pop, np.inf, dtype=float)
        self.owl_intensity = np.zeros(self.n_pop, dtype=float)
        self.owl_ic = np.zeros(self.n_pop, dtype=float)
        self.owl_r = np.zeros(self.n_pop, dtype=float)

        # best and weak solutions
        self.best_pos = None
        self.best_cost = float("inf")
        self.weak_pos = None
        self.weak_cost = -float("inf")

        # history
        self.best_per_iter: list[float] = []
        self.iter = 0

        # initialize population and evaluate
        self._init_population()

    def _safe_eval(self, x: np.ndarray) -> float:
        try:
            v = float(self.func(np.asarray(x, dtype=float)))
        except Exception as e:
            try:
                with open("failed_evals_osa.csv", "a", encoding="utf-8") as f:
                    f.write(f"{time.time()},{','.join(map(str, x))},exception,{str(e)}\n")
            except Exception:
                pass
            v = float("inf")
        if self.eval_delay:
            time.sleep(self.eval_delay)
        return v

    def _init_population(self) -> None:
        # uniform initialization in bounds
        for i in range(self.n_pop):
            self.owl_pos[i] = np.random.rand(self.dim) * (self.ub - self.lb) + self.lb
            self.owl_cost[i] = self._safe_eval(self.owl_pos[i])
            # track best and weak
            if self.owl_cost[i] < self.best_cost:
                self.best_cost = float(self.owl_cost[i])
                self.best_pos = self.owl_pos[i].copy()
            if self.owl_cost[i] > self.weak_cost:
                self.weak_cost = float(self.owl_cost[i])
                self.weak_pos = self.owl_pos[i].copy()
        # record initial best
        self.best_per_iter.append(self.best_cost)

    def _clip(self, x: np.ndarray) -> np.ndarray:
        return np.minimum(np.maximum(x, self.lb), self.ub)

    def save_checkpoint(self, path: Optional[str] = None) -> None:
        path = path or self.autosave_path
        if not path:
            return
        try:
            np.savez(
                path,
                owl_pos=self.owl_pos,
                owl_cost=self.owl_cost,
                best_pos=self.best_pos,
                best_cost=self.best_cost,
                weak_pos=self.weak_pos,
                weak_cost=self.weak_cost,
                iter=self.iter,
                best_per_iter=np.array(self.best_per_iter),
                lb=self.lb,
                ub=self.ub,
            )
        except Exception as e:
            try:
                with open("osa_save_error.log", "a", encoding="utf-8") as f:
                    f.write(f"{time.time()} save error: {e}\n")
                    f.write(traceback.format_exc())
            except Exception:
                pass

    def optimize(self) -> dict:
        """
        Run the Owl Search Algorithm main loop.
        Returns {"gbest","gbest_val","best_per_iter"}.
        """
        try:
            for t in range(1, self.max_iter + 1):
                self.iter = t
                pvm = float(np.random.rand())
                # decaying beta as in MATLAB: beta = 1.9 - 1.9 * (iter/maxiteration)
                beta = 1.9 - 1.9 * (t / float(self.max_iter))

                # select current best and weak
                # ensure we recompute best/weak from population (robust)
                mi = int(np.argmin(self.owl_cost))
                ma = int(np.argmax(self.owl_cost))
                self.best_cost = float(self.owl_cost[mi])
                self.best_pos = self.owl_pos[mi].copy()
                self.weak_cost = float(self.owl_cost[ma])
                self.weak_pos = self.owl_pos[ma].copy()

                # OWL Phase: update each owl
                for i in range(self.n_pop):
                    # Normalized intensity calculation (avoid div by zero)
                    denom = (self.weak_cost - self.best_cost)
                    if denom == 0:
                        self.owl_intensity[i] = 0.0
                    else:
                        self.owl_intensity[i] = (self.owl_cost[i] - self.best_cost) / denom

                    # Distance to best owl
                    self.owl_r[i] = np.linalg.norm(self.owl_pos[i] - self.best_pos)
                    # Update ic
                    self.owl_ic[i] = (self.owl_intensity[i] / (self.owl_r[i] ** 2 + np.finfo(float).eps)) + np.random.rand()

                    # Update position (two modes by pvm)
                    if pvm < 0.5:
                        newpos = self.owl_pos[i] + beta * self.owl_ic[i] * np.abs(alpha_value() * self.best_pos - self.owl_pos[i])
                    else:
                        newpos = self.owl_pos[i] - beta * self.owl_ic[i] * np.abs(alpha_value() * self.best_pos - self.owl_pos[i])

                    # Bound and evaluate
                    newpos = self._clip(newpos)
                    newcost = self._safe_eval(newpos)

                    # Greedy replacement
                    if newcost < self.owl_cost[i]:
                        self.owl_pos[i] = newpos
                        self.owl_cost[i] = newcost

                # update best/weak and history
                mi = int(np.argmin(self.owl_cost))
                ma = int(np.argmax(self.owl_cost))
                if self.owl_cost[mi] < self.best_cost:
                    self.best_cost = float(self.owl_cost[mi])
                    self.best_pos = self.owl_pos[mi].copy()
                if self.owl_cost[ma] > self.weak_cost:
                    self.weak_cost = float(self.owl_cost[ma])
                    self.weak_pos = self.owl_pos[ma].copy()

                self.best_per_iter.append(self.best_cost)

                # print progress occasionally
                if t % 10 == 0 or t == 1 or t == self.max_iter:
                    print(f"Iteration: {t} Best Cost: {self.best_cost:.6g}")

                # autosave
                if self.autosave_every_iters and (t % self.autosave_every_iters == 0):
                    self.save_checkpoint()

            # final autosave
            if self.autosave_every_iters:
                self.save_checkpoint()

            return {"gbest": self.best_pos, "gbest_val": float(self.best_cost), "best_per_iter": list(self.best_per_iter)}
        except Exception as e:
            try:
                with open("osa_exception.log", "a", encoding="utf-8") as f:
                    f.write(f"{time.time()} exception: {e}\n")
                    f.write(traceback.format_exc())
            except Exception:
                pass
            # try to save partial state
            try:
                if self.autosave_every_iters:
                    self.save_checkpoint((self.autosave_path or "osa_checkpoint.npz").replace(".npz", "_onexception.npz"))
            except Exception:
                pass
            raise

# helper: alpha in MATLAB code is randomized at start and used per iteration.
def alpha_value():
    # choose alpha uniformly in [0,0.5) as in OSA.m initial alpha = rand() * 0.5
    # Use a fixed per-call small random to keep behavior stochastic.
    return float(np.random.rand() * 0.5)

__all__ = ["OSA"]