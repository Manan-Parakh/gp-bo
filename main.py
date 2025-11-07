"""
gp_bo_rocket_demo.py

Toy demo: Gaussian Process surrogate + Bayesian Optimization (Expected Improvement)
for a synthetic rocket-engine-like thrust function with a safety constraint.

Author: ChatGPT (for Manan Parakh)
Date: 2025-11-06

Requirements:
  pip install numpy scipy scikit-learn matplotlib pandas pyDOE

Outputs:
  - gp_bo_results.csv : table of evaluated points and values
  - gp_bo_parity.png   : parity plot (true vs predicted)
  - gp_bo_ei_trace.png : BO convergence trace (best value vs iteration)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import time

# Optional: for Latin Hypercube Sampling
try:
    from pyDOE import lhs
except Exception:
    lhs = None
    # fallback LHS below if pyDOE not available

# -------------------------
# Synthetic functions
# -------------------------
def synthetic_thrust(X):
    """
    Synthetic thrust function (kN) depending on:
      X[:,0] = Pc (chamber pressure) in bar,
      X[:,1] = OF (oxidizer/fuel ratio)
    Returns: thrust in kN (float array)
    """
    Pc = X[:, 0]
    OF = X[:, 1]
    # plausible nonlinear combination (toy model)
    val = 0.5 * Pc * np.sqrt(OF) \
          + 0.1 * (Pc ** 1.5) * np.exp(-0.2 * (OF - 2.5) ** 2) \
          + 0.05 * Pc * OF
    return val

def synthetic_max_wall_temp(X):
    """
    Synthetic constraint function: maximum wall temperature (Celsius).
    We'll treat temperatures above a threshold as unsafe.
    This is intentionally correlated with Pc and OF in a nonlinear way.
    """
    Pc = X[:, 0]
    OF = X[:, 1]
    T = 300 + 0.4 * Pc + 20 * np.sin(0.15 * Pc) + 15 * (OF - 2.5) ** 2
    return T

# -------------------------
# Helper: Latin Hypercube Sampling (fallback)
# -------------------------
def simple_lhs(n_samples, n_dims, seed=None):
    rng = np.random.RandomState(seed)
    # create uniform grid and shuffle within each dimension
    result = np.zeros((n_samples, n_dims))
    for j in range(n_dims):
        perm = rng.permutation(n_samples)
        result[:, j] = (perm + rng.rand(n_samples)) / n_samples
    return result

# -------------------------
# Acquisition: Expected Improvement
# -------------------------
def expected_improvement(X_candidate, gp, y_best, xi=0.01):
    """
    X_candidate: shape (n_points, n_features)
    gp: fitted GaussianProcessRegressor
    y_best: current best observed objective (maximize)
    xi: exploration parameter
    Returns: EI values for each candidate
    """
    mu, sigma = gp.predict(X_candidate, return_std=True)
    sigma = np.maximum(sigma, 1e-9)
    # For maximization, improvement = mu - y_best - xi
    imp = mu - y_best - xi
    from scipy.stats import norm
    Z = imp / sigma
    ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei[sigma == 0.0] = 0.0
    return ei

# -------------------------
# Helper: maximize acquisition with multiple restarts
# -------------------------
def propose_location(acq_func, gp, y_best, bounds, n_restarts=10, seed=None):
    """
    Optimize acquisition function (maximize) via L-BFGS-B with random restarts.
    bounds: list of (low, high) for each dim
    Returns: best_x (1d array)
    """
    rng = np.random.RandomState(seed)
    dim = len(bounds)

    # objective for minimizer (we minimize negative acquisition)
    def min_obj(x):
        x = x.reshape(1, -1)
        return -acq_func(x, gp, y_best)

    best_x = None
    best_val = -np.inf
    for i in range(n_restarts):
        x0 = np.array([rng.uniform(b[0], b[1]) for b in bounds])
        res = minimize(min_obj, x0, bounds=bounds, method="L-BFGS-B",
                       options={"maxiter": 200})
        if not res.success:
            val = -min_obj(res.x)
        else:
            val = -res.fun
        if val > best_val:
            best_val = val
            best_x = res.x.copy()
    return np.clip(best_x, [b[0] for b in bounds], [b[1] for b in bounds])

# -------------------------
# Main BO workflow
# -------------------------
def run_bo(
    n_init=20,
    n_iter=20,
    noise_std=2.0,
    seed=42,
    pc_bounds=(20.0, 200.0),
    of_bounds=(1.5, 3.5),
    temp_limit=700.0,   # allowed max wall temperature (C)
):
    rng = np.random.RandomState(seed)

    bounds = [pc_bounds, of_bounds]
    dim = 2

    # 1) Initial samples via LHS
    n_init = int(n_init)
    if lhs is not None:
        unit = lhs(dim, samples=n_init, criterion='maximin')
    else:
        unit = simple_lhs(n_init, dim, seed=seed)

    # scale unit cube to bounds
    X_init = np.zeros((n_init, dim))
    for j in range(dim):
        X_init[:, j] = unit[:, j] * (bounds[j][1] - bounds[j][0]) + bounds[j][0]

    # true function + noise
    y_init = synthetic_thrust(X_init) + rng.normal(0, noise_std, n_init)
    temp_init = synthetic_max_wall_temp(X_init)

    # filter out initially unsafe points (simulate failed experiments)
    safe_mask = temp_init <= temp_limit
    if safe_mask.sum() == 0:
        # if none safe, relax and keep all but mark violations
        safe_mask = np.ones_like(safe_mask, dtype=bool)

    # We'll keep all points but mark safety status
    data = {
        "Pc": list(X_init[:, 0]),
        "OF": list(X_init[:, 1]),
        "thrust_kN": list(y_init),
        "temp_C": list(temp_init),
        "safe": list(temp_init <= temp_limit)
    }

    # Fit initial GP on entire initial set (including unsafe) but the BO will avoid unsafe when proposing
    X = np.array(data["Pc"])[:, None]
    X = np.hstack([np.array(data["Pc"])[:, None], np.array(data["OF"])[:, None]])
    y = np.array(data["thrust_kN"])

    # Standardize inputs for GP (benefits stability)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # GP kernel & model
    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=[50.0, 1.0], length_scale_bounds=(1e-2, 1e3))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=noise_std ** 2, n_restarts_optimizer=4, normalize_y=True, random_state=seed)
    gp.fit(X_scaled, y)

    # Logging
    best_so_far = []
    best_val = np.max(y)
    best_idx = np.argmax(y)
    best_point = X[best_idx, :]
    best_so_far.append(best_val)

    trace = []
    trace.append(best_val)

    # Begin BO iterations
    for it in range(n_iter):
        # propose a point via EI maximization BUT ensure candidate is likely safe
        y_best = np.max(y)
        # acquisition wrapper to operate on unscaled inputs
        def acq_on_unscaled(x_unscaled, gp_model, y_b):
            x_unscaled = np.atleast_2d(x_unscaled)
            x_scaled = scaler.transform(x_unscaled)
            return expected_improvement(x_scaled, gp_model, y_b, xi=0.01)[0]

        # propose location (unscaled)
        candidate = propose_location(lambda Xc, g, yb: acq_on_unscaled(Xc, g, yb),
                                     gp, y_best, bounds, n_restarts=12, seed=seed + it)

        # Evaluate safety first (simulated)
        candidate_arr = np.atleast_2d(candidate)
        cand_temp = synthetic_max_wall_temp(candidate_arr)[0]
        is_safe = cand_temp <= temp_limit

        # If unsafe, attempt a few random-safe proposals before accepting an unsafe point (safe-BO heuristic)
        safety_tries = 0
        while (not is_safe) and safety_tries < 6:
            # try a random candidate sampled from safe-biased distribution
            random_c = np.array([rng.uniform(b[0], b[1]) for b in bounds])
            cand_temp = synthetic_max_wall_temp(random_c.reshape(1, -1))[0]
            if cand_temp <= temp_limit:
                candidate = random_c
                is_safe = True
                break
            safety_tries += 1

        # Evaluate objective (expensive eval) with noise to simulate measurement/simulation error
        val = synthetic_thrust(candidate_arr)[0] + rng.normal(0, noise_std)
        temp_val = synthetic_max_wall_temp(candidate_arr)[0]

        # Append to dataset
        X = np.vstack([X, candidate_arr])
        y = np.hstack([y, val])

        data["Pc"].append(candidate[0])
        data["OF"].append(candidate[1])
        data["thrust_kN"].append(val)
        data["temp_C"].append(temp_val)
        data["safe"].append(bool(temp_val <= temp_limit))

        # Refit GP on scaled data (fit on all observed data)
        X_scaled = scaler.fit_transform(X)
        gp.fit(X_scaled, y)

        # update best
        if val > best_val and (temp_val <= temp_limit):
            best_val = val
            best_point = candidate.copy()
        trace.append(best_val)

        # print progress
        print(f"Iter {it+1:02d}/{n_iter:02d} | cand Pc={candidate[0]:6.2f} bar, OF={candidate[1]:4.2f}, thrust={val:7.3f} kN, temp={temp_val:6.2f} C, safe={temp_val <= temp_limit}")

    # Summarize and save results
    df = pd.DataFrame(data)
    df["iteration"] = range(1, len(df) + 1)
    output_dir = "gp_bo_outputs"
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "gp_bo_results.csv")
    df.to_csv(csv_path, index=False)

    print("\n=== BO complete ===")
    print(f"Best safe thrust observed: {best_val:.4f} kN at Pc={best_point[0]:.2f}, OF={best_point[1]:.2f}")
    print(f"Saved results to {csv_path}")

    # Return artifacts
    return gp, scaler, X, y, df, trace, best_val, best_point, output_dir

# -------------------------
# Plotting utilities
# -------------------------
def plot_parity_and_uncertainty(gp, scaler, X_obs, y_obs, output_dir):
    # build test grid and show GP predictive mean vs true
    pc_lin = np.linspace(20, 200, 60)
    of_lin = np.linspace(1.5, 3.5, 40)
    PC, OF = np.meshgrid(pc_lin, of_lin)
    pts = np.vstack([PC.ravel(), OF.ravel()]).T
    pts_scaled = scaler.transform(pts)
    mu, sigma = gp.predict(pts_scaled, return_std=True)
    mu = mu.reshape(OF.shape)
    sigma = sigma.reshape(OF.shape)

    # plot mean surface (contour)
    plt.figure(figsize=(10, 5))
    cs = plt.contourf(PC, OF, mu, levels=20)
    plt.colorbar(cs, label="Predicted thrust (kN)")
    plt.scatter(X_obs[:, 0], X_obs[:, 1], c='k', s=12, label="evaluations")
    plt.xlabel("Pc (bar)")
    plt.ylabel("O/F")
    plt.title("GP Predicted Thrust Surface (kN) with Observations")
    plt.legend()
    plt.tight_layout()
    fpath = os.path.join(output_dir, "gp_bo_pred_surface.png")
    plt.savefig(fpath, dpi=200)
    plt.close()

def plot_bo_trace(trace, output_dir):
    plt.figure(figsize=(6,4))
    plt.plot(range(len(trace)), trace, marker='o')
    plt.xlabel("Iteration (including inits)")
    plt.ylabel("Best observed thrust (kN)")
    plt.title("BO Convergence Trace")
    plt.grid(True)
    plt.tight_layout()
    fpath = os.path.join(output_dir, "gp_bo_ei_trace.png")
    plt.savefig(fpath, dpi=200)
    plt.close()

def plot_parity(gp, scaler, X_obs, y_obs, output_dir):
    # parity: predicted vs true on observed points via LOOCV-like predict
    X_scaled = scaler.transform(X_obs)
    y_pred, y_std = gp.predict(X_scaled, return_std=True)
    plt.figure(figsize=(6,6))
    plt.errorbar(y_obs, y_pred, y_std, fmt='o', alpha=0.7)
    mn = min(y_obs.min(), y_pred.min())
    mx = max(y_obs.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx], linestyle='--')
    plt.xlabel("Observed thrust (kN)")
    plt.ylabel("GP predicted thrust (kN)")
    plt.title("Parity plot (with predictive std)")
    plt.grid(True)
    plt.tight_layout()
    fpath = os.path.join(output_dir, "gp_bo_parity.png")
    plt.savefig(fpath, dpi=200)
    plt.close()

# -------------------------
# Entrypoint
# -------------------------
if __name__ == "__main__":
    t0 = time.time()
    gp, scaler, X_obs, y_obs, df, trace, best_val, best_point, outdir = run_bo(
        n_init=30,
        n_iter=25,
        noise_std=2.0,
        seed=2025,
        pc_bounds=(20.0, 200.0),
        of_bounds=(1.6, 3.4),
        temp_limit=720.0
    )
    print("Plotting results...")
    plot_parity_and_uncertainty(gp, scaler, X_obs, y_obs, outdir)
    plot_bo_trace(trace, outdir)
    plot_parity(gp, scaler, X_obs, y_obs, outdir)
    print(f"Saved plots in {outdir}: gp_bo_pred_surface.png, gp_bo_ei_trace.png, gp_bo_parity.png")
    print(f"Total time: {time.time() - t0:.1f} s")
