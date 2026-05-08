"""
RRT-Based Optimizer (RRTO) - Python translation of RTO.m

Interface compatible with other optimizers in this repo:

Usage:
    rto = RTO(func, pop, dim, lb, ub, max_iter,
              autosave_every_iters=0, autosave_path="rto_checkpoint.npz", eval_delay=0.0, rng_seed=None)
    res = rto.optimize()
    res -> {"gbest": array, "gbest_val": float, "best_per_iter": list}

Notes:
- func(x) should accept a 1D numpy array and return a scalar objective value.
- lb/ub may be scalars or length-D iterables. They will be expanded to length D arrays.
- This implementation follows the MATLAB RTO.m structure but includes safe evaluation,
  optional eval_delay, autosave/checkpointing and some small numerical guards.
"""
import numpy as np
import time
import traceback
import os

class RTO:
    def __init__(self, func, pop, dim, lb, ub, max_iter,
                 autosave_every_iters=0, autosave_path="rto_checkpoint.npz", eval_delay=0.0, rng_seed=None):
        self.func = func
        self.pop = int(pop)
        self.dim = int(dim)
        # allow scalar lb/ub or sequences
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

        # initialization
        self.X = self._initialize_population(self.pop, self.dim, self.ub, self.lb)
        self.curr_score = np.full(self.pop, np.inf)
        for i in range(self.pop):
            self.curr_score[i] = self._safe_eval(self.X[i])
        best_idx = int(np.argmin(self.curr_score))
        self.best_pos = self.X[best_idx].copy()
        self.best_val = float(self.curr_score[best_idx])

        # bookkeeping
        self.best_per_iter = []
        self.iter = 0

    def _safe_eval(self, x):
        try:
            v = float(self.func(np.array(x, dtype=float)))
        except Exception as e:
            # log failed eval and return +inf so optimizer can continue
            try:
                with open("failed_evals_rto.csv", "a", encoding="utf-8") as f:
                    f.write(f"{time.time()},{','.join(map(str, x))},exception,{str(e)}\n")
            except Exception:
                pass
            v = float("inf")
        if self.eval_delay:
            time.sleep(self.eval_delay)
        return v

    def _initialize_population(self, N, D, ub, lb):
        # handle vector bounds
        ub = np.array(ub, dtype=float)
        lb = np.array(lb, dtype=float)
        Pos = np.random.rand(N, D) * (ub - lb) + lb
        return Pos

    def save_checkpoint(self, path=None):
        if path is None:
            path = self.autosave_path
        try:
            np.savez(path,
                     X=self.X,
                     curr_score=self.curr_score,
                     best_pos=self.best_pos,
                     best_val=self.best_val,
                     iter=self.iter,
                     best_per_iter=np.array(self.best_per_iter),
                     lb=self.lb,
                     ub=self.ub)
        except Exception as e:
            try:
                with open("rto_save_error.log", "a", encoding="utf-8") as f:
                    f.write(f"{time.time()} save error: {e}\n")
                    f.write(traceback.format_exc())
            except Exception:
                pass

    def _boundary_handle_scalar(self, idx, dim_idx):
        # fallback boundary handling (used if out-of-bounds)
        # similar to MATLAB: if dim > 15 choose gene from other individual else random within bounds
        if self.dim > 15:
            sel = np.random.randint(0, self.pop - 1)
            if sel >= idx:
                sel += 1
            return self.X[sel, dim_idx]
        else:
            return np.random.rand() * (self.ub[dim_idx] - self.lb[dim_idx]) + self.lb[dim_idx]

    def optimize(self):
        try:
            N = self.pop
            D = self.dim
            MaxIter = self.max_iter
            Pop = self.X.copy()
            Currentscore = self.curr_score.copy()
            Bestposition = self.best_pos.copy()
            Bestscore = float(self.best_val)

            Convergence_curve = np.full(MaxIter, np.inf, dtype=float)

            it = 1
            C = 10.0  # penalty factor used in step sizing in MATLAB
            # main loop (MATLAB used 1..MaxIter inclusive)
            while it <= MaxIter:
                # compute adaptive coefficients (guard against log(0))
                if MaxIter - it > 0:
                    k = np.log(MaxIter - it) / (np.log(MaxIter) + np.finfo(float).eps)
                else:
                    k = 0.0
                E = (it / MaxIter) ** (1.0 / 3.0)
                m1 = E / 10.0
                m2 = E / 50.0

                newpop = Pop.copy()

                # per-individual, per-dimension update
                for i in range(N):
                    for j in range(D):
                        # adaptive step size wandering strategy (r1)
                        r1 = np.random.rand()
                        if r1 < k:
                            S1 = (r1 - (k / 2.0)) * k * (self.ub[j] - self.lb[j]) / C
                            newpop[i, j] = Pop[i, j] + S1

                        # absolute difference-based adaptive step size strategy (r2)
                        r2 = np.random.rand()
                        if r2 < m1:
                            # b = exp(cos(pi*(1-(1/it))));  but for it=1 this is safe
                            b = np.exp(np.cos(np.pi * (1.0 - (1.0 / max(1, it)))))
                            alpha1 = 5.0 * (r2 - m1 / 2.0) * np.cos(2.0 * np.pi * r2) * np.exp(b)
                            # S2 is vector = alpha1 * abs(Best_j - Pop_i_vector)
                            S2 = alpha1 * np.abs(Bestposition[j] - Pop[i, :])
                            # assign full-dimension vector to newpop row (MATLAB did newpop(i,:)=...)
                            newpop[i, :] = Bestposition[j] + S2

                        # boundary-based adaptive step size strategy (r3)
                        r3 = np.random.rand()
                        if r3 < m2:
                            beta = 10.0 * np.pi * it / MaxIter
                            alpha2 = r3 * (r3 - m2 / 2.0) * k * (1.0 - it / MaxIter)
                            S3 = (self.ub[j] - self.lb[j]) * np.cos(beta) * alpha2
                            # MATLAB did newpop(i,j)=Bestposition(1,j)+S3
                            newpop[i, j] = Bestposition[j] + S3

                # boundary/collision handling and evaluation
                for i in range(N):
                    # enforce bounds (reflecting as in MATLAB: replace out-of-bounds with ub or lb)
                    C_ub = newpop[i, :] > self.ub
                    C_lb = newpop[i, :] < self.lb
                    # combine: if >ub set ub, if <lb set lb, else keep value
                    newpop[i, C_ub] = self.ub[C_ub]
                    newpop[i, C_lb] = self.lb[C_lb]
                    # evaluate
                    newscore_i = self._safe_eval(newpop[i, :])
                    # update if improvement
                    if newscore_i < Currentscore[i]:
                        Currentscore[i] = newscore_i
                        Pop[i, :] = newpop[i, :].copy()
                        if newscore_i < Bestscore:
                            Bestscore = float(newscore_i)
                            Bestposition = Pop[i, :].copy()

                Convergence_curve[it - 1] = Bestscore
                self.best_per_iter.append(Bestscore)
                self.iter = it

                # optional autosave
                if self.autosave_every_iters and (it % self.autosave_every_iters == 0):
                    self.X = Pop.copy()
                    self.curr_score = Currentscore.copy()
                    self.best_pos = Bestposition.copy()
                    self.best_val = Bestscore
                    self.save_checkpoint()

                # print status occasionally
                if it % 10 == 0 or it == 1 or it == MaxIter:
                    print(f"Iteration: {it} Best Cost = {Bestscore:.6g}")

                it += 1

            # finalize state
            self.X = Pop.copy()
            self.curr_score = Currentscore.copy()
            self.best_pos = Bestposition.copy()
            self.best_val = Bestscore

            # final autosave
            if self.autosave_every_iters:
                self.save_checkpoint()

            return {"gbest": self.best_pos, "gbest_val": self.best_val, "best_per_iter": list(self.best_per_iter)}
        except Exception as e:
            try:
                with open("rto_exception.log", "a", encoding="utf-8") as f:
                    f.write(f"{time.time()} exception: {e}\n")
                    f.write(traceback.format_exc())
            except Exception:
                pass
            try:
                self.save_checkpoint(self.autosave_path.replace(".npz", "_onexception.npz"))
            except Exception:
                pass
            raise