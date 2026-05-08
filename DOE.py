"""
Dhole Optimization (DOE) - Python translation of DOE.m

Usage:
    from DOE import DOE
    doe = DOE(func, pop, dim, lb, ub, max_iter,
              autosave_every_iters=0, autosave_path="doe_checkpoint.npz", eval_delay=0.0, rng_seed=None)
    res = doe.optimize()
    res -> {"gbest": array, "gbest_val": float, "best_per_iter": list}

Notes:
- func(x) should accept a 1D numpy array and return a scalar objective.
- lb/ub may be scalars or length-d iterables. They will be expanded to length dim arrays.
- This implementation follows the structure of the provided MATLAB DOE (dhole) function.
- Adds safe evaluation (exceptions -> +inf), optional eval_delay, autosave/checkpointing.
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

class DOE:
    def __init__(
        self,
        func: Callable[[np.ndarray], float],
        pop: int,
        dim: int,
        lb,
        ub,
        max_iter: int,
        autosave_every_iters: int = 0,
        autosave_path: str = "doe_checkpoint.npz",
        eval_delay: float = 0.0,
        rng_seed: Optional[int] = None,
    ):
        self.func = func
        self.pop = _to_int_scalar(pop, "pop")
        self.dim = _to_int_scalar(dim, "dim")
        self.max_iter = _to_int_scalar(max_iter, "max_iter")

        # Bounds handling: scalar or vector
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

        # state
        self.X = self._initialization(self.pop, self.dim, self.ub, self.lb)
        self.fitness = np.full(self.pop, np.inf, dtype=float)
        self.best_per_iter: list[float] = []
        self.iter = 0

        # evaluate initial population and set best
        best_val = float("inf")
        best_pos = None
        for i in range(self.pop):
            self.fitness[i] = self._safe_eval(self.X[i, :])
            if self.fitness[i] < best_val:
                best_val = float(self.fitness[i])
                best_pos = self.X[i, :].copy()
        self.prey_global = best_pos.copy() if best_pos is not None else np.zeros(self.dim)
        self.best_val = best_val
        self.best_per_iter.append(self.best_val)

    def _safe_eval(self, x: np.ndarray) -> float:
        try:
            v = float(self.func(np.asarray(x, dtype=float)))
        except Exception as e:
            try:
                with open("failed_evals_doe.csv", "a", encoding="utf-8") as f:
                    f.write(f"{time.time()},{','.join(map(str, x))},exception,{str(e)}\n")
            except Exception:
                pass
            v = float("inf")
        if self.eval_delay:
            time.sleep(self.eval_delay)
        return v

    @staticmethod
    def _p_obj(x):
        """
        p_obj(x) from MATLAB:
        y = ((1 / (1 + exp(-0.5 * (x- 25))))^2)* rand;
        Expect x as scalar (PWN) in MATLAB; here we mimic scalar behavior.
        """
        # allow vector or scalar; apply elementwise and return scalar random-weighted value
        xv = np.asarray(x)
        # compute sigmoid-like component
        val = 1.0 / (1.0 + np.exp(-0.5 * (xv - 25.0)))
        # square and multiply by random scalar in (0,1)
        return (val ** 2) * np.random.rand()

    @staticmethod
    def _initialization(SearchAgents_no, dim, ub, lb):
        """
        Initialize population analogously to MATLAB initialization() helper.
        ub and lb may be arrays or scalars already passed as arrays.
        """
        ub = np.asarray(ub, dtype=float)
        lb = np.asarray(lb, dtype=float)
        # If ub and lb are scalars (length 1), treat as uniform bounds
        if ub.size == 1:
            return np.random.rand(SearchAgents_no, dim) * (ub - lb) + lb
        else:
            P = np.zeros((SearchAgents_no, dim), dtype=float)
            for j in range(dim):
                ub_j = ub[j]
                lb_j = lb[j]
                P[:, j] = np.random.rand(SearchAgents_no) * (ub_j - lb_j) + lb_j
            return P

    def _clip(self, x: np.ndarray) -> np.ndarray:
        return np.minimum(np.maximum(x, self.lb), self.ub)

    def save_checkpoint(self, path: Optional[str] = None) -> None:
        path = path or self.autosave_path
        if not path:
            return
        try:
            np.savez(
                path,
                X=self.X,
                fitness=self.fitness,
                prey_global=self.prey_global,
                best_val=self.best_val,
                iter=self.iter,
                best_per_iter=np.array(self.best_per_iter),
                lb=self.lb,
                ub=self.ub,
            )
        except Exception as e:
            try:
                with open("doe_save_error.log", "a", encoding="utf-8") as f:
                    f.write(f"{time.time()} save error: {e}\n")
                    f.write(traceback.format_exc())
            except Exception:
                pass

    def optimize(self) -> dict:
        """
        Run Dhole (DOE) optimization for self.max_iter iterations.
        Returns dict {"gbest","gbest_val","best_per_iter"}.
        """
        try:
            T = self.max_iter
            N = self.pop
            dim = self.dim

            # local copies as in MATLAB
            X = self.X.copy()
            fitness_f = self.fitness.copy()
            prey_global = self.prey_global.copy()
            localBest_position = prey_global.copy()
            Best_fitness = self.best_val

            cuve_f = np.zeros(T, dtype=float)
            cuve_f[0] = Best_fitness
            t = 1

            # ensure Xnew allocated
            Xnew = X.copy()

            while t <= T:
                C = 1.0 - (t / float(T))
                PWN = int(np.round(np.random.rand() * 15 + 5))  # int in MATLAB
                prey = 0.5 * (prey_global + localBest_position)
                prey_local = localBest_position.copy()

                # generate Xnew for each individual
                for i in range(N):
                    if np.random.rand() < 0.5:
                        if PWN < 10:
                            # Searching stage
                            Xnew[i, :] = X[i, :] + C * np.random.rand() * (prey - X[i, :])
                        else:
                            # Encircling stage
                            # For each dimension choose z random index != i
                            for j in range(dim):
                                z = np.random.randint(0, N)
                                while z == i:
                                    z = np.random.randint(0, N)
                                Xnew[i, j] = X[i, j] - X[z, j] + prey[j]
                    else:
                        # Hunting stage
                        denom = self._safe_eval(prey_local)
                        # protect denominator from zero/infinite
                        if not np.isfinite(denom) or denom == 0:
                            denom = 1e-12
                        Q = 3.0 * np.random.rand() * fitness_f[i] / denom
                        if Q > 2.0:
                            W_prey = np.exp(-1.0 / Q) * prey_local
                            # p_obj uses scalar PWN; produce a scalar multiplier
                            p_scalar = DOE._p_obj(PWN)
                            for j in range(dim):
                                Xnew[i, j] = (
                                    X[i, j]
                                    + np.cos(2.0 * np.pi * np.random.rand()) * W_prey[j] * p_scalar
                                    - np.sin(2.0 * np.pi * np.random.rand()) * W_prey[j] * p_scalar
                                )
                        else:
                            p_scalar = DOE._p_obj(PWN)
                            Xnew[i, :] = (X[i, :] - prey_global) * p_scalar + p_scalar * np.random.rand(dim) * X[i, :]

                # boundary conditions
                # clip Xnew to bounds
                for i in range(N):
                    Xnew[i, :] = self._clip(Xnew[i, :])

                # local best initialization based on Xnew[0]
                localBest_position = Xnew[0, :].copy()
                localBest_fitness = self._safe_eval(localBest_position)

                # evaluate new population and update
                for i in range(N):
                    local_fitness = self._safe_eval(Xnew[i, :])
                    if local_fitness < localBest_fitness:
                        localBest_fitness = local_fitness
                        localBest_position = Xnew[i, :].copy()
                    if local_fitness < fitness_f[i]:
                        fitness_f[i] = local_fitness
                        X[i, :] = Xnew[i, :].copy()
                        if fitness_f[i] < Best_fitness:
                            Best_fitness = float(fitness_f[i])
                            prey_global = X[i, :].copy()

                cuve_f[t - 1] = Best_fitness
                # store trajectory element (optional)
                # T_particle(t)= Xnew(1); in matlab Xnew(1) ambiguous — skip
                t += 1

                # record iteration counter
                self.iter = t - 1
                self.best_per_iter.append(Best_fitness)

                if (self.iter % 10 == 0) or (self.iter == 1) or (self.iter == T):
                    print(f"DOE iter {self.iter}/{T}: Best = {Best_fitness:.6g}")

                if self.autosave_every_iters and (self.iter % self.autosave_every_iters == 0):
                    # save current state
                    self.X = X.copy()
                    self.fitness = fitness_f.copy()
                    self.prey_global = prey_global.copy()
                    self.best_val = Best_fitness
                    self.save_checkpoint()

            # finish
            self.X = X.copy()
            self.fitness = fitness_f.copy()
            self.prey_global = prey_global.copy()
            self.best_val = Best_fitness

            return {"gbest": self.prey_global, "gbest_val": float(self.best_val), "best_per_iter": list(self.best_per_iter)}
        except Exception as e:
            try:
                with open("doe_exception.log", "a", encoding="utf-8") as f:
                    f.write(f"{time.time()} exception: {e}\n")
                    f.write(traceback.format_exc())
            except Exception:
                pass
            try:
                if self.autosave_every_iters:
                    self.save_checkpoint((self.autosave_path or "doe_checkpoint.npz").replace(".npz", "_onexception.npz"))
            except Exception:
                pass
            raise