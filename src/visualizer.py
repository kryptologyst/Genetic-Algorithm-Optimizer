import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
from loguru import logger


class GAVisualizer:
    @staticmethod
    def plot_convergence(history, save_path=None):
        plt.figure(figsize=(8, 5))
        plt.plot(history, linewidth=2, color="seagreen")
        plt.xlabel("Generation"); plt.ylabel("Best Fitness")
        plt.title("Genetic Algorithm Convergence")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
