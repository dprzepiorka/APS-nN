import numpy as np
import time
import traceback
import os

class PSO:
    """
    PSO optimizer similar in interface to CEO.
    Usage:
        pso = PSO(func, n_particles, dim, lb, ub, max_iter, w, c1, c2,
                  autosave_every_iters=5, autosave_path="pso_checkpoint.npz", eval_delay=0.0)
        res = pso.optimize()
    Returns dict: {"gbest": ..., "gbest_val": ..., "best_per_iter": [...]}
    """
    def __init__(self, func, n_particles, dim, lb, ub, max_iter, w=0.7, c1=1.5, c2=1.5,
                 autosave_every_iters=0, autosave_path="pso_checkpoint.npz", eval_delay=0.0):
        self.func = func
        self.n_particles = int(n_particles)
        self.dim = int(dim)
        self.lb = np.array(lb, dtype=float)
        self.ub = np.array(ub, dtype=float)
        self.max_iter = int(max_iter)
        self.w = float(w)
        self.c1 = float(c1)
        self.c2 = float(c2)

        self.autosave_every_iters = int(autosave_every_iters)
        self.autosave_path = autosave_path
        self.eval_delay = float(eval_delay)

        # initialize swarm
        self.X = np.random.uniform(self.lb, self.ub, (self.n_particles, self.dim))
        self.V = np.zeros_like(self.X)
        self.pbest = self.X.copy()
        self.pbest_val = np.full(self.n_particles, np.inf)
        self.gbest = None
        self.gbest_val = np.inf
        self.best_per_iter = []

        self.iter = 0

    def _eval_particle(self, x, p_idx=None):
        try:
            val = float(self.func(x))
        except Exception as e:
            # log failed evaluation
            try:
                with open("failed_evals_pso.csv", "a", encoding="utf-8") as f:
                    f.write(f"{time.time()},{','.join(map(str, x))},exception,{str(e)}\n")
            except Exception:
                pass
            val = np.inf
        if self.eval_delay:
            time.sleep(self.eval_delay)
        return val

    def save_checkpoint(self, path=None):
        if path is None:
            path = self.autosave_path
        try:
            np.savez(path,
                     X=self.X,
                     V=self.V,
                     pbest=self.pbest,
                     pbest_val=self.pbest_val,
                     gbest=self.gbest,
                     gbest_val=self.gbest_val,
                     best_per_iter=np.array(self.best_per_iter),
                     iter=self.iter,
                     lb=self.lb,
                     ub=self.ub)
            # human-readable history
            with open(os.path.splitext(path)[0] + "_history.txt", "w", encoding="utf-8") as f:
                for el in self.best_per_iter:
                    f.write(f"{el}\n")
        except Exception as e:
            try:
                with open("pso_save_error.log", "a", encoding="utf-8") as f:
                    f.write(f"{time.time()} save error: {e}\n")
                    f.write(traceback.format_exc())
            except Exception:
                pass

    def optimize(self):
        try:
            # initial eval
            for p in range(self.n_particles):
                val = self._eval_particle(self.X[p], p)
                self.pbest_val[p] = val
                if val < self.gbest_val:
                    self.gbest_val = val
                    self.gbest = self.X[p].copy()
            self.best_per_iter.append(self.gbest_val)
            # main loop
            for it in range(1, self.max_iter + 1):
                self.iter = it
                for p in range(self.n_particles):
                    r1 = np.random.rand(self.dim)
                    r2 = np.random.rand(self.dim)
                    self.V[p] = self.w * self.V[p] + self.c1 * r1 * (self.pbest[p] - self.X[p]) + self.c2 * r2 * (self.gbest - self.X[p])
                    vmax = (self.ub - self.lb) * 0.5
                    self.V[p] = np.clip(self.V[p], -vmax, vmax)
                    self.X[p] = np.clip(self.X[p] + self.V[p], self.lb, self.ub)

                    val = self._eval_particle(self.X[p], p)
                    if val < self.pbest_val[p]:
                        self.pbest_val[p] = val
                        self.pbest[p] = self.X[p].copy()
                    if val < self.gbest_val:
                        self.gbest_val = val
                        self.gbest = self.X[p].copy()

                self.best_per_iter.append(self.gbest_val)

                if self.autosave_every_iters and (it % self.autosave_every_iters == 0):
                    self.save_checkpoint()

            # final save
            if self.autosave_every_iters:
                self.save_checkpoint()
            return {"gbest": self.gbest, "gbest_val": self.gbest_val, "best_per_iter": self.best_per_iter}
        except Exception as e:
            # save on exception
            try:
                with open("pso_exception.log", "a", encoding="utf-8") as f:
                    f.write(f"{time.time()} exception: {e}\n")
                    f.write(traceback.format_exc())
            except Exception:
                pass
            try:
                self.save_checkpoint(self.autosave_path.replace(".npz", "_onexception.npz"))
            except Exception:
                pass
            raise

# convenience function for backwards compatibility if someone wants to import run_pso
def run_pso_with_obj(func, vars_def_or_dim, lb, ub, n_particles, n_iter, w, c1, c2, autosave_every=0, autosave_path="pso_checkpoint.npz", eval_delay=0.0):
    """
    Helper if you want to call PSO directly with a function:
      func(x) -> objective
    vars_def_or_dim: can be integer dim or list-like of variable defs (we only need dim)
    """
    if isinstance(vars_def_or_dim, int):
        dim = vars_def_or_dim
    else:
        dim = len(vars_def_or_dim)
    pso = PSO(func, n_particles, dim, lb, ub, n_iter, w, c1, c2, autosave_every, autosave_path, eval_delay)
    return pso.optimize()