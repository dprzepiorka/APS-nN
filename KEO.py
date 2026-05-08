"""
Kangaroo Escape Optimizer (KEO) - Python implementation translated from KEO.m

Interface consistent with CEO/PSO classes in this repository.

Usage (CEO-compatible signature):
    keo = KEO(func, n_kangaroos, dim, lb, ub, N, max_fes,
              autosave_every_iters=0, autosave_path="keo_checkpoint.npz",
              eval_delay=0.0, rng_seed=None, initial_solutions=None)
    res = keo.optimize()
    res -> {"gbest": array, "gbest_val": float, "best_per_iter": list}

Notes:
- func(x) should accept a 1D numpy array and return scalar objective.
- lb/ub may be scalars or length-d iterables. They will be expanded to length dim arrays.
- initial_solutions (optional) should be an array-like of shape (n_kangaroos, dim).
- The constructor signature is kept similar to CEO so it can be used interchangeably.
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

class KEO:
    def __init__(
        self,
        func: Callable[[np.ndarray], float],
        numKangaroos,
        dim,
        lb,
        ub,
        N=None,
        max_fes=None,
        autosave_every_iters: int = 0,
        autosave_path: str = "keo_checkpoint.npz",
        eval_delay: float = 0.0,
        rng_seed: Optional[int] = None,
        initial_solutions: Optional[np.ndarray] = None,
        group_fraction: float = 0.05,
        energy_threshold: float = 0.5
    ):
        """
        Keep constructor similar to CEO: (func, n_agents, dim, lb, ub, N, max_fes, ...)

        - N and max_fes are accepted for API compatibility (N can be used to pass other param)
        - initial_solutions: optional array shape (numKangaroos, dim). If None, random init used.
        """
        self.func = func
        self.numKangaroos = _to_int_scalar(numKangaroos, "numKangaroos")
        self.dim = _to_int_scalar(dim, "dim")
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
                raise ValueError(f"lb must be scalar or length-{self.dim} array, got shape {arr_lb.shape}")

        if np.isscalar(ub):
            self.ub = np.full(self.dim, float(ub))
        else:
            arr_ub = np.asarray(ub, dtype=float)
            if arr_ub.size == 1:
                self.ub = np.full(self.dim, float(arr_ub.item()))
            elif arr_ub.size == self.dim:
                self.ub = arr_ub.copy()
            else:
                raise ValueError(f"ub must be scalar or length-{self.dim} array, got shape {arr_ub.shape}")

        if np.any(self.ub <= self.lb):
            raise ValueError("Each ub must be greater than lb for all dimensions")

        self.N = N
        self.max_fes = max_fes

        self.autosave_every_iters = int(autosave_every_iters)
        self.autosave_path = autosave_path if autosave_every_iters else None
        self.eval_delay = float(eval_delay)
        if rng_seed is not None:
            np.random.seed(int(rng_seed))

        # hyperparams derived from MATLAB code
        self.Group_Size = max(1, int(round(group_fraction * self.numKangaroos)))
        self.EnergyThreshold = float(energy_threshold)

        # state arrays
        if initial_solutions is not None:
            arr = np.asarray(initial_solutions, dtype=float)
            if arr.shape[0] != self.numKangaroos or arr.shape[1] != self.dim:
                raise ValueError("initial_solutions must have shape (numKangaroos, dim)")
            self.Kangeroo = arr.copy()
        else:
            self.Kangeroo = np.random.rand(self.numKangaroos, self.dim) * (self.ub - self.lb) + self.lb

        self.KangerooFit = np.full(self.numKangaroos, np.inf, dtype=float)
        self.Decoy_Drop = np.zeros((self.numKangaroos, self.dim), dtype=float)

        # initialize fitness and best
        for i in range(self.numKangaroos):
            self.KangerooFit[i] = self._safe_eval(self.Kangeroo[i, :])
        bidx = int(np.argmin(self.KangerooFit))
        self.BestF = float(self.KangerooFit[bidx])
        self.BestX = self.Kangeroo[bidx, :].copy()

        # chaotic_val init as in MATLAB
        self.chaotic_val = 0.7

        # history
        self.best_per_iter: list[float] = [self.BestF]
        self.iter = 0

    def _safe_eval(self, x: np.ndarray) -> float:
        try:
            v = float(self.func(np.asarray(x, dtype=float)))
        except Exception as e:
            try:
                with open("failed_evals_keo.csv", "a", encoding="utf-8") as f:
                    f.write(f"{time.time()},{','.join(map(str, x.tolist()))},exception,{str(e)}\n")
            except Exception:
                pass
            v = float("inf")
        if self.eval_delay:
            time.sleep(self.eval_delay)
        return v

    @staticmethod
    def _zigzag_escape(current: np.ndarray, best: np.ndarray, beta1: float, theta_max_deg: float) -> np.ndarray:
        """
        Zig-zag escape: perturb the current position with angular-like variation relative to best.
        Implementation: create a perturbation vector whose magnitude is proportional to (current-best)
        and whose direction is randomized within theta_max_deg (element-wise random).
        """
        delta = current - best
        norm_delta = np.linalg.norm(delta) + 1e-12
        # generate per-dimension angles in radians within [-theta_max, theta_max]
        th = np.deg2rad(theta_max_deg)
        angles = np.random.uniform(-th, th, size=delta.shape)
        # create rotated-like perturbation: element-wise scale by cos(angle) and add small orthogonal noise
        orth_noise = np.random.randn(*delta.shape) * 0.1 * (np.abs(delta) + 1e-6)
        perturb = (np.cos(angles) * delta) + (np.sin(angles) * orth_noise)
        new = current + beta1 * perturb * (0.5 + 0.5 * np.random.rand())  # scale
        return new

    def _clip(self, x: np.ndarray) -> np.ndarray:
        return np.minimum(np.maximum(x, self.lb), self.ub)

    def save_checkpoint(self, path: Optional[str] = None) -> None:
        path = path or self.autosave_path
        if not path:
            return
        try:
            np.savez(
                path,
                Kangeroo=self.Kangeroo,
                KangerooFit=self.KangerooFit,
                BestX=self.BestX,
                BestF=self.BestF,
                iter=self.iter,
                best_per_iter=np.array(self.best_per_iter),
                lb=self.lb,
                ub=self.ub,
            )
        except Exception as e:
            try:
                with open("keo_save_error.log", "a", encoding="utf-8") as f:
                    f.write(f"{time.time()} save error: {e}\n")
                    f.write(traceback.format_exc())
            except Exception:
                pass

    def optimize(self, max_iter: Optional[int] = None) -> dict:
        """
        Run the KEO algorithm. If max_iter is provided it overrides the MaxIt passed (if any).
        Returns {"gbest", "gbest_val", "best_per_iter"}.
        """
        try:
            MaxIt = int(max_iter) if max_iter is not None else (int(self.N) if self.N is not None else 100)
            # if user provided max_fes as a fallback, use it to set MaxIt if N not provided
            if MaxIt <= 0:
                MaxIt = 100

            for It in range(1, MaxIt + 1):
                self.iter = It
                # Reset Decoy_Drop for this iteration (MATLAB initializes inside loop)
                self.Decoy_Drop[:] = 0.0

                # update chaotic_val using logistic map (MATLAB logic)
                if It == 1:
                    self.chaotic_val = 0.7
                else:
                    # logistic map: x_{n+1} = 4 x_n (1 - x_n)
                    self.chaotic_val = 4.0 * self.chaotic_val * (1.0 - self.chaotic_val)

                for ii in range(self.numKangaroos):
                    r = np.random.rand()
                    if np.random.rand() > 0.5:
                        # Stage 1: Escape Updating (long jump or zigzag)
                        Energy_Level = (1.0 - np.random.rand() * (It / float(MaxIt))) * (0.95 + 0.05 * self.chaotic_val)

                        # second random comparison rand > rand in MATLAB: emulate by comparing two independent randoms
                        if Energy_Level > self.EnergyThreshold and (np.random.rand() > np.random.rand()):
                            # Stage 1.1 Long Jump Escape Mechanism & false move (Decoy_Drop currently zeros initially)
                            # Use gaussian jump scaled by current kangaroo and decoy drop mask
                            # If Decoy_Drop is zero (early iterations), fallback to random jump
                            dd = self.Decoy_Drop[ii, :]
                            # If dd is all zeros, create a small random mask to allow motion
                            if not np.any(dd):
                                dd = np.random.choice([0.0, 1.0], size=self.dim, p=[0.7, 0.3])
                            Jump = 2.0 * np.random.randn() * dd * self.Kangeroo[ii, :]
                            newKangeroo = self.Kangeroo[ii, :] + Jump
                        else:
                            # Stage 1.2 Zig-Zag Escape Mechanism
                            beta1 = 1.0
                            theta_max_deg = 30.0
                            newKangeroo = KEO._zigzag_escape(self.Kangeroo[ii, :], self.BestX, beta1, theta_max_deg)

                    else:
                        # Stage 2: Escape Updating by seeking Safer_Areas
                        if r < 1.0 / 3.0:
                            self.Decoy_Drop[ii, :] = 1.0
                        elif r < 2.0 / 3.0:
                            self.Decoy_Drop[ii, :] = np.round(np.random.rand(self.dim))
                        else:
                            self.Decoy_Drop[ii, :] = np.round(np.random.rand(self.dim) * np.random.rand(self.dim))

                        # choose safer area
                        if It < 2 * MaxIt / 4.0 or (np.random.rand() > np.random.rand()):
                            Safer_Area = np.random.randint(0, self.numKangaroos)
                        else:
                            if np.random.rand() < 0.75:
                                # pick random group and choose best among them
                                Safe_group = np.random.randint(0, self.numKangaroos, size=self.Group_Size)
                                # ensure unique indices
                                Safe_group = np.unique(Safe_group)
                                # if group smaller than 1, ensure one index
                                if Safe_group.size == 0:
                                    Safe_group = np.array([np.random.randint(0, self.numKangaroos)])
                                sg_fits = self.KangerooFit[Safe_group]
                                Selected_one = int(np.argmin(sg_fits))
                                Safer_Area = int(Safe_group[Selected_one])
                            else:
                                Safer_Area = int(np.argmin(self.KangerooFit))

                        # Update position towards safer area
                        # Use normal noise scaled by Decoy_Drop elementwise
                        newKangeroo = self.Kangeroo[Safer_Area, :] + np.random.randn() * self.Decoy_Drop[ii, :] * (
                            self.Kangeroo[ii, :] - self.Kangeroo[Safer_Area, :]
                        )

                    # Bound checks
                    newKangeroo = self._clip(newKangeroo)

                    # Evaluate fitness
                    newKangerooFit = self._safe_eval(newKangeroo)

                    # Accept if better
                    if newKangerooFit < self.KangerooFit[ii]:
                        self.KangerooFit[ii] = newKangerooFit
                        self.Kangeroo[ii, :] = newKangeroo

                # update global best
                cur_best_idx = int(np.argmin(self.KangerooFit))
                if self.KangerooFit[cur_best_idx] < self.BestF:
                    self.BestF = float(self.KangerooFit[cur_best_idx])
                    self.BestX = self.Kangeroo[cur_best_idx, :].copy()

                # record history
                self.best_per_iter.append(self.BestF)

                # progress print
                if It % 10 == 0 or It == 1 or It == MaxIt:
                    print(f"KEO Iter {It}/{MaxIt} Best = {self.BestF:.6g}")

                # autosave
                if self.autosave_every_iters and (It % self.autosave_every_iters == 0):
                    self.save_checkpoint()

            # final autosave
            if self.autosave_every_iters:
                self.save_checkpoint()

            return {"gbest": self.BestX, "gbest_val": self.BestF, "best_per_iter": list(self.best_per_iter)}
        except Exception as e:
            try:
                with open("keo_exception.log", "a", encoding="utf-8") as f:
                    f.write(f"{time.time()} exception: {e}\n")
                    f.write(traceback.format_exc())
            except Exception:
                pass
            # try to save partial state
            try:
                if self.autosave_every_iters:
                    self.save_checkpoint((self.autosave_path or "keo_checkpoint.npz").replace(".npz", "_onexception.npz"))
            except Exception:
                pass
            raise

# allow "from KEO import KEO"
__all__ = ["KEO"]