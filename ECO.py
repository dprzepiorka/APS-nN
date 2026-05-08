"""
Ecological Cycle Optimizer (ECO) - Python implementation translated from ECO.m

Usage:
    from ECO import ECO
    eco = ECO(func, pop_size, dim, lb, ub, max_iter,
              autosave_every_iters=0, autosave_path="eco_checkpoint.npz",
              eval_delay=0.0, rng_seed=None)
    res = eco.optimize()
    res -> {"gbest": array, "gbest_val": float, "best_per_iter": list}

Notes:
- func(x) must accept a 1D numpy array and return a scalar objective.
- lb/ub may be scalars or length-d iterables. They will be expanded to length dim arrays.
- This module adds safe evaluations, autosave/checkpointing and a history similar to other optimizers.
"""
from __future__ import annotations
import numpy as np
import time
import traceback
import os
from typing import Callable, Optional

_EPS = np.finfo(float).eps


def _to_int_scalar(x, name="value"):
    if np.isscalar(x):
        return int(x)
    arr = np.asarray(x)
    if arr.size == 1:
        return int(arr.item())
    raise TypeError(f"'{name}' must be a scalar or 1-element array/list, got shape {arr.shape}")


class ECO:
    def __init__(
        self,
        func: Callable[[np.ndarray], float],
        pop_size,
        dim,
        lb,
        ub,
        max_iter,
        autosave_every_iters: int = 0,
        autosave_path: str = "eco_checkpoint.npz",
        eval_delay: float = 0.0,
        rng_seed: Optional[int] = None,
    ):
        self.func = func
        self.pop_size = _to_int_scalar(pop_size, "pop_size")
        self.dim = _to_int_scalar(dim, "dim")
        self.max_iter = _to_int_scalar(max_iter, "max_iter")

        # bounds handling
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

        self.autosave_every_iters = int(autosave_every_iters)
        self.autosave_path = autosave_path if autosave_every_iters else None
        self.eval_delay = float(eval_delay)
        if rng_seed is not None:
            np.random.seed(int(rng_seed))

        # population proportions (from MATLAB)
        self.P_producer = 0.2
        self.P_herbivore = 0.3
        self.P_carnivore = 0.3
        # omnivores remainder

        self.Pro_num = max(1, int(round(self.pop_size * self.P_producer)))
        self.Her_num = max(1, int(round(self.pop_size * self.P_herbivore)))
        self.Car_num = max(1, int(round(self.pop_size * self.P_carnivore)))
        self.Omn_num = max(1, self.pop_size - self.Pro_num - self.Her_num - self.Car_num)
        self.Dec_num = self.pop_size

        # state
        self.best_per_iter: list[float] = []
        self.iter = 0

        # initialize population
        self.Pop_pos = np.zeros((self.pop_size, self.dim), dtype=float)
        self.Pop_fit = np.full(self.pop_size, np.inf, dtype=float)
        for i in range(self.pop_size):
            self.Pop_pos[i, :] = self.lb + (self.ub - self.lb) * np.random.rand(self.dim)
            self.Pop_fit[i] = self._safe_eval(self.Pop_pos[i, :])

        best_idx = int(np.argmin(self.Pop_fit))
        self.Best_fit = float(self.Pop_fit[best_idx])
        self.Best_pos = self.Pop_pos[best_idx, :].copy()
        self.best_per_iter.append(self.Best_fit)

        # initialize subgroups
        # ensure we have correct slicing even if sums differ slightly
        idx = 0
        self.Producer_pos = self.Pop_pos[idx: idx + self.Pro_num, :].copy(); idx += self.Pro_num
        self.Producer_fit = self.Pop_fit[0: self.Pro_num].copy()
        self.Herbivore_pos = self.Pop_pos[idx: idx + self.Her_num, :].copy(); idx += self.Her_num
        self.Herbivore_fit = self.Pop_fit[self.Pro_num: self.Pro_num + self.Her_num].copy()
        self.Carnivore_pos = self.Pop_pos[idx: idx + self.Car_num, :].copy(); idx += self.Car_num
        self.Carnivore_fit = self.Pop_fit[self.Pro_num + self.Her_num: self.Pro_num + self.Her_num + self.Car_num].copy()
        self.Omnivore_pos = self.Pop_pos[idx: idx + self.Omn_num, :].copy()
        start_omn = self.Pro_num + self.Her_num + self.Car_num
        self.Omnivore_fit = self.Pop_fit[start_omn: start_omn + self.Omn_num].copy()

    def _safe_eval(self, x: np.ndarray) -> float:
        try:
            val = float(self.func(np.asarray(x, dtype=float)))
        except Exception as e:
            try:
                with open("failed_evals_eco.csv", "a", encoding="utf-8") as f:
                    f.write(f"{time.time()},{','.join(map(str, x.tolist()))},exception,{str(e)}\n")
            except Exception:
                pass
            val = float("inf")
        if self.eval_delay:
            time.sleep(self.eval_delay)
        return val

    @staticmethod
    def _boundmapping(pos: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
        # replace out-of-bounds variables with random in-bounds values at those positions
        pos = np.asarray(pos, dtype=float).copy()
        flag_upper = pos > ub
        flag_lower = pos < lb
        flag = flag_upper | flag_lower
        if np.any(flag):
            rand_vals = lb + (ub - lb) * np.random.rand(pos.size)
            pos = np.where(flag, rand_vals, pos)
        return pos

    @staticmethod
    def _pos_update(pos_old: np.ndarray, fit_old: float, pos_new: np.ndarray, best_pos: np.ndarray, best_fit: float, fobj: Callable[[np.ndarray], float], eval_fn: Callable[[np.ndarray], float]) -> tuple:
        pos_new = np.asarray(pos_new, dtype=float)
        new_fit = eval_fn(pos_new)
        if new_fit < fit_old:
            pos = pos_new
            fit = new_fit
            if new_fit < best_fit:
                best_pos = pos_new.copy()
                best_fit = new_fit
        else:
            pos = pos_old
            fit = fit_old
        return pos, fit, best_pos, best_fit

    @staticmethod
    def _roulette(fit: np.ndarray, k: int):
        # roulette selection based on 1/(fit+eps) probabilities; returns indices (0-based), sampling with replacement
        fit = np.asarray(fit, dtype=float)
        # handle infinite or invalid fitness => set large value so prob small
        mask_inf = ~np.isfinite(fit)
        if np.all(mask_inf):
            # fallback to random indices
            return np.random.randint(0, fit.size, size=k)
        safe_fit = fit.copy()
        safe_fit[mask_inf] = np.max(safe_fit[~mask_inf]) + 1.0 if np.any(~mask_inf) else 1.0
        inv = 1.0 / (safe_fit + _EPS)
        probs = inv / np.sum(inv)
        cum = np.cumsum(probs)
        r = np.random.rand(k)
        idx = np.searchsorted(cum, r)
        # clip indices
        idx = np.clip(idx, 0, fit.size - 1)
        return idx

    def save_checkpoint(self, path: Optional[str] = None):
        path = path or self.autosave_path
        if not path:
            return
        try:
            np.savez(
                path,
                Best_pos=self.Best_pos,
                Best_fit=self.Best_fit,
                Producer_pos=self.Producer_pos,
                Producer_fit=self.Producer_fit,
                Herbivore_pos=self.Herbivore_pos,
                Herbivore_fit=self.Herbivore_fit,
                Carnivore_pos=self.Carnivore_pos,
                Carnivore_fit=self.Carnivore_fit,
                Omnivore_pos=self.Omnivore_pos,
                Omnivore_fit=self.Omnivore_fit,
                iter=self.iter,
                best_per_iter=np.array(self.best_per_iter),
                lb=self.lb,
                ub=self.ub,
            )
        except Exception as e:
            try:
                with open("eco_save_error.log", "a", encoding="utf-8") as f:
                    f.write(f"{time.time()} save error: {e}\n")
                    f.write(traceback.format_exc())
            except Exception:
                pass

    def optimize(self):
        try:
            Max_it = int(self.max_iter)
            for It in range(1, Max_it + 1):
                self.iter = It
                # update predation coefficient G (per-dimension)
                G = np.ones(self.dim, dtype=float)
                # sign +/-1 randomly per dimension
                signs = np.random.choice([-1.0, 1.0], size=self.dim)
                decay = np.exp(-9.0 * (It / float(Max_it)) ** 3)
                G = 1.0 + 2.0 * np.random.rand(self.dim) * decay * signs

                # (1) Producer strategy: sort hybrid of decomposers+producers if not first iteration
                if It != 1:
                    # ensure Decomposer exists
                    Decomposer_pos = np.vstack([self.Producer_pos, self.Herbivore_pos, self.Carnivore_pos, self.Omnivore_pos])
                    Decomposer_fit = np.hstack([self.Producer_fit, self.Herbivore_fit, self.Carnivore_fit, self.Omnivore_fit])
                    order = np.argsort(Decomposer_fit)
                    pos_hybrid = Decomposer_pos[order]
                    fit_hybrid = Decomposer_fit[order]
                    self.Producer_pos = pos_hybrid[: self.Pro_num].copy()
                    self.Producer_fit = fit_hybrid[: self.Pro_num].copy()
                    if self.Producer_fit.size > 0 and self.Producer_fit[0] < self.Best_fit:
                        self.Best_pos = self.Producer_pos[0].copy()
                        self.Best_fit = float(self.Producer_fit[0])

                # (2) Herbivore strategy
                if self.Producer_fit.size >= 1 and self.Her_num > 0:
                    r1 = self._roulette(self.Producer_fit, min(3, self.Producer_fit.size))
                    for i in range(self.Her_num):
                        # if fewer than 3 indices, sample with replacement
                        while r1.size < 3:
                            r1 = np.concatenate([r1, self._roulette(self.Producer_fit, 1)])
                        r1 = r1[:3]
                        Her_new = self.Herbivore_pos[i, :] + G * (
                            np.random.rand() * (self.Producer_pos[r1[0], :] - self.Herbivore_pos[i, :])
                            + np.random.rand() * (self.Producer_pos[r1[1], :] - self.Herbivore_pos[i, :])
                            + np.random.rand() * (self.Producer_pos[r1[2], :] - self.Herbivore_pos[i, :])
                        )
                        Her_new = self._boundmapping(Her_new, self.lb, self.ub)
                        (self.Herbivore_pos[i, :], self.Herbivore_fit[i], self.Best_pos, self.Best_fit) = self._pos_update(
                            self.Herbivore_pos[i, :],
                            self.Herbivore_fit[i] if i < self.Herbivore_fit.size else float("inf"),
                            Her_new,
                            self.Best_pos,
                            self.Best_fit,
                            self.func,
                            self._safe_eval,
                        )

                # (3) Carnivore strategy
                if self.Herbivore_fit.size >= 1 and self.Car_num > 0:
                    r2 = self._roulette(self.Herbivore_fit, min(3, self.Herbivore_fit.size))
                    for i in range(self.Car_num):
                        while r2.size < 3:
                            r2 = np.concatenate([r2, self._roulette(self.Herbivore_fit, 1)])
                        r2 = r2[:3]
                        Car_new = self.Carnivore_pos[i, :] + G * (
                            np.random.rand() * (self.Herbivore_pos[r2[0], :] - self.Carnivore_pos[i, :])
                            + np.random.rand() * (self.Herbivore_pos[r2[1], :] - self.Carnivore_pos[i, :])
                            + np.random.rand() * (self.Herbivore_pos[r2[2], :] - self.Carnivore_pos[i, :])
                        )
                        Car_new = self._boundmapping(Car_new, self.lb, self.ub)
                        (self.Carnivore_pos[i, :], self.Carnivore_fit[i], self.Best_pos, self.Best_fit) = self._pos_update(
                            self.Carnivore_pos[i, :],
                            self.Carnivore_fit[i] if i < self.Carnivore_fit.size else float("inf"),
                            Car_new,
                            self.Best_pos,
                            self.Best_fit,
                            self.func,
                            self._safe_eval,
                        )

                # (4) Omnivore strategy
                # sample indices
                if (self.Producer_fit.size >= 1 or self.Herbivore_fit.size >= 1 or self.Carnivore_fit.size >= 1) and self.Omn_num > 0:
                    r3 = self._roulette(self.Producer_fit, 1) if self.Producer_fit.size >= 1 else np.array([0])
                    r4 = self._roulette(self.Herbivore_fit, 1) if self.Herbivore_fit.size >= 1 else np.array([0])
                    r5 = self._roulette(self.Carnivore_fit, min(2, max(1, self.Carnivore_fit.size))) if self.Carnivore_fit.size >= 1 else np.array([0, 0])
                    # ensure sizes
                    if r5.size < 2:
                        more = self._roulette(self.Carnivore_fit, 2 - r5.size) if self.Carnivore_fit.size >= 1 else np.random.randint(0, self.Car_num, size=(2 - r5.size))
                        r5 = np.concatenate([r5, more])
                    for i in range(self.Omn_num):
                        # pick scalar indices
                        ir3 = int(r3[0]) if r3.size > 0 else 0
                        ir4 = int(r4[0]) if r4.size > 0 else 0
                        ir51 = int(r5[0])
                        ir52 = int(r5[1])
                        Omn_new = self.Omnivore_pos[i, :] + G * (
                            np.random.rand() * (self.Producer_pos[ir3, :] - self.Omnivore_pos[i, :])
                            + np.random.rand() * (self.Herbivore_pos[ir4, :] - self.Omnivore_pos[i, :])
                            + np.random.rand() * (self.Carnivore_pos[ir51, :] - self.Omnivore_pos[i, :])
                            + np.random.rand() * (self.Carnivore_pos[ir52, :] - self.Omnivore_pos[i, :])
                        )
                        Omn_new = self._boundmapping(Omn_new, self.lb, self.ub)
                        (self.Omnivore_pos[i, :], self.Omnivore_fit[i], self.Best_pos, self.Best_fit) = self._pos_update(
                            self.Omnivore_pos[i, :],
                            self.Omnivore_fit[i] if i < self.Omnivore_fit.size else float("inf"),
                            Omn_new,
                            self.Best_pos,
                            self.Best_fit,
                            self.func,
                            self._safe_eval,
                        )

                # (5) Decomposer strategy
                Decomposer_pos = np.vstack([self.Producer_pos, self.Herbivore_pos, self.Carnivore_pos, self.Omnivore_pos])
                Decomposer_fit = np.hstack([self.Producer_fit, self.Herbivore_fit, self.Carnivore_fit, self.Omnivore_fit])
                # ensure lengths
                if Decomposer_fit.size == 0:
                    min_num = 0
                else:
                    min_num = int(np.argmin(Decomposer_fit))
                # ensure Decomposer arrays defined
                Dec_pos = Decomposer_pos.copy()
                Dec_fit = Decomposer_fit.copy()
                for i in range(self.Dec_num):
                    if np.random.rand() < 0.5:
                        # Optimal decomposition
                        Bestpos_neihood = Dec_pos[min_num, :] * np.random.rand(self.dim)
                        Dec_new = Bestpos_neihood + (0.4 * np.random.rand() - 0.2) * (Bestpos_neihood - Dec_pos[i % Dec_pos.shape[0], :])
                    else:
                        if np.random.rand() < 0.5:
                            # Local random decomposition
                            dis = np.linalg.norm(Dec_pos[min_num, :] - Dec_pos[i % Dec_pos.shape[0], :])
                            rand_vec = 2.0 * np.random.rand(self.dim) - 1.0
                            norm_rand = rand_vec / ( _EPS + np.linalg.norm(rand_vec))
                            Dec_new = Dec_pos[i % Dec_pos.shape[0], :] + np.random.rand() * dis * norm_rand
                        else:
                            # Global random decomposition
                            H = (1.0 - It / (1.5 * Max_it)) ** (5.0 * It / Max_it) * (np.cos(np.pi * np.random.rand()))
                            # min(Low-Up) in MATLAB corresponds to min(lb-ub) negative; use min(ub-lb)
                            rand_walk = 2.0 / 3.0 * H * np.random.rand() * np.min(self.ub - self.lb)
                            weight = np.random.rand()
                            Dec_new = weight * Dec_pos[i % Dec_pos.shape[0], :] + (1.0 - weight) * rand_walk
                    Dec_new = self._boundmapping(Dec_new, self.lb, self.ub)
                    # evaluate and update
                    # ensure indexing within arrays
                    target_idx = i % Dec_pos.shape[0]
                    old_pos = Dec_pos[target_idx, :].copy()
                    old_fit = Dec_fit[target_idx] if target_idx < Dec_fit.size else float("inf")
                    (new_pos, new_fit, _, _) = self._pos_update(old_pos, old_fit, Dec_new, self.Best_pos, self.Best_fit, self.func, self._safe_eval)
                    Dec_pos[target_idx, :] = new_pos
                    Dec_fit[target_idx] = new_fit
                    # update global best if changed inside _pos_update
                    if new_fit < self.Best_fit:
                        self.Best_fit = new_fit
                        self.Best_pos = new_pos.copy()

                # After decomposer loop, optionally reassign subgroup arrays from Dec_pos/Dec_fit
                # Re-split Decomposer into groups order as earlier (Producer, Herbivore, Carnivore, Omnivore)
                total = Dec_pos.shape[0]
                # ensure sizes consistent
                p_end = min(self.Pro_num, total)
                h_end = min(self.Pro_num + self.Her_num, total)
                c_end = min(self.Pro_num + self.Her_num + self.Car_num, total)
                o_end = min(self.Pro_num + self.Her_num + self.Car_num + self.Omn_num, total)
                if total >= p_end:
                    self.Producer_pos = Dec_pos[0:p_end, :].copy()
                    self.Producer_fit = Dec_fit[0:p_end].copy()
                if total >= h_end:
                    self.Herbivore_pos = Dec_pos[p_end:h_end, :].copy()
                    self.Herbivore_fit = Dec_fit[p_end:h_end].copy()
                if total >= c_end:
                    self.Carnivore_pos = Dec_pos[h_end:c_end, :].copy()
                    self.Carnivore_fit = Dec_fit[h_end:c_end].copy()
                if total >= o_end:
                    self.Omnivore_pos = Dec_pos[c_end:o_end, :].copy()
                    self.Omnivore_fit = Dec_fit[c_end:o_end].copy()

                # Save the best value for the current iteration
                self.best_per_iter.append(float(self.Best_fit))

                # print progress occasionally
                if It % 10 == 0 or It == 1 or It == Max_it:
                    print(f"ECO Iter {It}/{Max_it} Best = {self.Best_fit:.6g}")

                # autosave
                if self.autosave_every_iters and (It % self.autosave_every_iters == 0):
                    self.save_checkpoint()

            # final autosave
            if self.autosave_every_iters:
                self.save_checkpoint()

            return {"gbest": self.Best_pos, "gbest_val": float(self.Best_fit), "best_per_iter": list(self.best_per_iter)}
        except Exception as e:
            try:
                with open("eco_exception.log", "a", encoding="utf-8") as f:
                    f.write(f"{time.time()} exception: {e}\n")
                    f.write(traceback.format_exc())
            except Exception:
                pass
            try:
                if self.autosave_every_iters:
                    self.save_checkpoint((self.autosave_path or "eco_checkpoint.npz").replace(".npz", "_onexception.npz"))
            except Exception:
                pass
            raise


__all__ = ["ECO"]