\
from __future__ import annotations
from typing import List
import numpy as np
import matplotlib.pyplot as plt

def plot_fit_svg(x: List[float], y: List[float], alpha: float, beta: float, title: str, out_path: str) -> None:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    y_fit = alpha + beta * x_arr
    plt.figure(figsize=(10, 6))
    plt.scatter(x_arr, y_arr, label="measured")
    plt.plot(x_arr, y_fit, label="fit")
    plt.xlabel("Bytes (B)", fontsize=16)
    plt.ylabel("Latency (ms)", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # Intentionally omit title to keep plots clean per user request
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, format="pdf")
    plt.close()
