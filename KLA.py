"""
Kirchhoff's Law Algorithm (KLA) - Python translation of KLA.m

Usage:
    kla = KLA(func, n_pop, dim, lb, ub, max_evals,
              autosave_every_iters=0, autosave_path="kla_checkpoint.npz",
              eval_delay=0.0, rng_seed=None)
    res = kla.optimize()
    res -> {"gbest": array, "gbest_val": float, "best_per_iter": list}

Notes:
- func(x) must accept a 1D numpy array and return a scalar objective.
- lb/ub may be scalars or length-D iterables. They will be expanded to length D arrays.
- max_evals follows the MATLAB code semantics: counts function evaluations. The algorithm
  initializes the population (n_pop evaluations) and then generates one candidate per
  individual per loop, so evaluations grow by n_pop per outer loop iteration.
- The implementation includes safe evaluation (exceptions -> +inf), optional eval_delay,
  and autosave/checkpointing.
"""
import numpy as np
import time
import traceback
import os

class KLA:
    def __init__(self, func, n_pop, dim, lb, ub, max_evals,
                 autosave_every_iters=0, autosave_path="kla_checkpoint.npz",
                 eval_delay=0.0, rng_seed=None):
        self.func = func
        self.n_pop = int(n_pop)
        self.dim = int(dim)

        # support scalar or vector bounds
        if np.isscalar(lb):
            self.lb = np.full(self.dim, float(lb))
        else:
            self.lb = np.array(lb, dtype=float)
        if np.isscalar(ub):
            self.ub = np.full(self.dim, float(ub))
        else:
            self.ub = np.array(ub, dtype=float)

        self.max_evals = int(max_evals)
        self.autosave_every_iters = int(autosave_every_iters)
        self.autosave_path = autosave_path
        self.eval_delay = float(eval_delay)

        if rng_seed is not None:
            np.random.seed(rng_seed)

        # tiny epsilon like MATLAB realmin
        self._eps = np.finfo(float).tiny

        # initialize population and bookkeeping
        self.Pop = []  # list of dicts { "X": arr, "Cost": float }
        self.best_per_iter = []
        self.iter_eval = 0  # counts evaluations
        self._init_population()

    def _safe_eval(self, x):
        try:
            v = float(self.func(np.array(x, dtype=float)))
        except Exception as e:
            # log failed eval and return +inf
            try:
                with open("failed_evals_kla.csv", "a", encoding="utf-8") as f:
                    f.write(f"{time.time()},{','.join(map(str, x))},exception,{str(e)}\n")
            except Exception:
                pass
            v = float("inf")
        if self.eval_delay:
            time.sleep(self.eval_delay)
        return v

    def _init_population(self):
        # create initial population and evaluate each individual
        self.Pop = []
        for i in range(self.n_pop):
            x = np.random.uniform(self.lb, self.ub, self.dim)
            c = self._safe_eval(x)
            self.Pop.append({"X": x, "Cost": c})
            self.iter_eval += 1
        # find best
        costs = [p["Cost"] for p in self.Pop]
        idx = int(np.argmin(costs))
        self.Best = self.Pop[idx].copy()
        # store initial best
        self.best_per_iter.append(self.Best["Cost"])

    def save_checkpoint(self, path=None):
        path = path or self.autosave_path
        try:
            np.savez(path,
                     Pop_X=np.array([p['X'] for p in self.Pop]),
                     Pop_C=np.array([p['Cost'] for p in self.Pop]),
                     Best_X=self.Best['X'],
                     Best_C=self.Best['Cost'],
                     iter_eval=self.iter_eval,
                     best_per_iter=np.array(self.best_per_iter),
                     lb=self.lb,
                     ub=self.ub)
        except Exception as e:
            try:
                with open("kla_save_error.log", "a", encoding="utf-8") as f:
                    f.write(f"{time.time()} save error: {e}\n")
                    f.write(traceback.format_exc())
            except Exception:
                pass

    def _clip(self, x):
        return np.minimum(np.maximum(x, self.lb), self.ub)

    def optimize(self):
        """
        Run optimization until self.iter_eval >= self.max_evals.
        Returns dict {"gbest","gbest_val","best_per_iter"}.
        """
        try:
            nPop = self.n_pop
            VarMin = self.lb
            VarMax = self.ub
            MaxEvals = self.max_evals
            eps = self._eps

            # If initial population already met or exceeded max_evals, return current best.
            if self.iter_eval >= MaxEvals:
                return {"gbest": self.Best["X"], "gbest_val": self.Best["Cost"], "best_per_iter": list(self.best_per_iter)}

            # Main loop: each loop generates nPop candidate evaluations (one per individual)
            while self.iter_eval < MaxEvals:
                # for each individual produce a candidate
                for i in range(nPop):
                    # prepare random indices excluding i
                    A = np.random.permutation(nPop)
                    # remove occurrence of i in A (if present)
                    A = A[A != i]
                    # ensure we have at least 3 distinct indices
                    if A.size < 3:
                        # fill with random indices (allow repeats) if necessary
                        extra = np.random.randint(0, nPop, size=(3 - A.size))
                        A = np.concatenate([A, extra])
                    a = int(A[0])
                    b = int(A[1])
                    jj = int(A[2])

                    Pi = self.Pop[i]
                    P_a = self.Pop[a]
                    P_b = self.Pop[b]
                    P_jj = self.Pop[jj]

                    # compute q, Q, Q2 safely (add eps to denominators)
                    denom_jj = (abs(Pi["Cost"] - P_jj["Cost"]) + eps)
                    q = ((Pi["Cost"] - P_jj["Cost"]) + eps) / denom_jj

                    denom_a = (abs(Pi["Cost"] - P_a["Cost"]) + eps)
                    Q = (Pi["Cost"] - P_a["Cost"]) / denom_a

                    denom_b = (abs(Pi["Cost"] - P_b["Cost"]) + eps)
                    Q2 = (Pi["Cost"] - P_b["Cost"]) / denom_b

                    # ratios for exponential factors (guard division by zero)
                    # Use positive denominators by adding eps and absolute to avoid complex exponents
                    safe_Pi_cost = Pi["Cost"] if abs(Pi["Cost"]) > eps else eps
                    safe_jj_cost = P_jj["Cost"] if abs(P_jj["Cost"]) > eps else eps
                    safe_a_cost = P_a["Cost"] if abs(P_a["Cost"]) > eps else eps
                    safe_b_cost = P_b["Cost"] if abs(P_b["Cost"]) > eps else eps

                    # these mimic MATLAB power operations; ensure non-negative base for real exponent by using ratio of magnitudes
                    try:
                        q1 = (abs(safe_jj_cost) / abs(safe_Pi_cost)) ** (2.0 * np.random.rand())
                    except Exception:
                        q1 = 1.0
                    try:
                        Q1 = (abs(safe_a_cost) / abs(safe_Pi_cost)) ** (2.0 * np.random.rand())
                    except Exception:
                        Q1 = 1.0
                    try:
                        Q21 = (abs(safe_b_cost) / abs(safe_Pi_cost)) ** (2.0 * np.random.rand())
                    except Exception:
                        Q21 = 1.0

                    # compute S1,S2,S3
                    rand_vec = np.random.rand(self.dim)
                    S1 = q1 * q * rand_vec * (P_jj["X"] - Pi["X"])
                    S2 = Q * Q1 * np.random.rand(self.dim) * (P_a["X"] - Pi["X"])
                    S3 = Q2 * Q21 * np.random.rand(self.dim) * (P_b["X"] - Pi["X"])

                    # scalar multipliers for each component
                    s1_mul = (np.random.rand() + np.random.rand())
                    s2_mul = (np.random.rand() + np.random.rand())
                    s3_mul = (np.random.rand() + np.random.rand())

                    S = s1_mul * S1 + s2_mul * S2 + s3_mul * S3

                    newX = Pi["X"] + S
                    newX = self._clip(newX)

                    # evaluate new candidate
                    newC = self._safe_eval(newX)
                    self.iter_eval += 1

                    # accept if improved (<= as in original)
                    if newC <= Pi["Cost"]:
                        # update individual's solution
                        self.Pop[i] = {"X": newX, "Cost": newC}
                        # update best if necessary
                        if newC <= self.Best["Cost"]:
                            self.Best = {"X": newX.copy(), "Cost": newC}

                    # store best-per-eval history (like BestCost1 in MATLAB)
                    self.best_per_iter.append(self.Best["Cost"])

                    # check termination
                    if self.iter_eval >= MaxEvals:
                        break

                # optionally autosave after each outer pass
                if self.autosave_every_iters and (self.iter_eval % (self.autosave_every_iters) == 0):
                    self.save_checkpoint()

                # break outer while if reached
                if self.iter_eval >= MaxEvals:
                    break

            # final checkpoint
            if self.autosave_every_iters:
                self.save_checkpoint()

            return {"gbest": np.array(self.Best["X"]), "gbest_val": float(self.Best["Cost"]), "best_per_iter": list(self.best_per_iter)}
        except Exception as e:
            try:
                with open("kla_exception.log", "a", encoding="utf-8") as f:
                    f.write(f"{time.time()} exception: {e}\n")
                    f.write(traceback.format_exc())
            except Exception:
                pass
            try:
                # try to save whatever state we have
                self.save_checkpoint((self.autosave_path or "kla_checkpoint.npz").replace(".npz", "_onexception.npz"))
            except Exception:
                pass
            raise

# Expose class for from KLA import KLA
__all__ = ["KLA"]