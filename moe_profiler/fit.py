\
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

@dataclass
class FitResult:
    alpha: float
    beta: float
    r2: float

def linear_fit_alpha_beta(x_bytes: List[float], y_ms: List[float]) -> FitResult:
    """
    Fit y = alpha + beta * x using least squares, return alpha, beta, R^2.
    """
    x = np.asarray(x_bytes, dtype=np.float64)
    y = np.asarray(y_ms, dtype=np.float64)
    if x.size < 2:
        return FitResult(alpha=float("nan"), beta=float("nan"), r2=float("nan"))
    X = np.vstack([np.ones_like(x), x]).T  # [N,2]
    theta, *_ = np.linalg.lstsq(X, y, rcond=None)
    alpha, beta = theta.tolist()
    # R^2
    y_pred = X @ theta
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    return FitResult(alpha=alpha, beta=beta, r2=r2)

def grid_fit_per_edge(meas: Dict[Tuple[int,int], Dict[str, List[Tuple[float,float]]]],
                      key: str) -> Tuple[List[List[float]], List[List[float]], List[List[float]]]:
    """
    meas[(i,j)][key] -> list of (bytes, ms), return alpha[i][j], beta[i][j], r2[i][j]
    """
    # find max indices
    max_i = max((i for (i,_) in meas.keys()), default=-1)
    max_j = max((j for (_,j) in meas.keys()), default=-1)
    I = max_i + 1
    J = max_j + 1
    alpha = [[0.0 for _ in range(J)] for __ in range(I)]
    beta  = [[0.0 for _ in range(J)] for __ in range(I)]
    r2    = [[1.0 for _ in range(J)] for __ in range(I)]
    for (i,j), d in meas.items():
        pairs = d.get(key, [])
        bytes_list = [b for (b, ms) in pairs]
        ms_list = [ms for (b, ms) in pairs]
        fr = linear_fit_alpha_beta(bytes_list, ms_list)
        alpha[i][j] = float(fr.alpha) if fr.alpha==fr.alpha else 0.0  # NaN -> 0
        beta[i][j]  = float(fr.beta)  if fr.beta==fr.beta   else 0.0
        r2[i][j]    = float(fr.r2)    if fr.r2==fr.r2       else 0.0
    return alpha, beta, r2
