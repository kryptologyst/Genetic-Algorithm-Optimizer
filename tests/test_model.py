import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import GeneticAlgorithm
from src.data import rastrigin_function


class TestGeneticAlgorithm:
    def test_optimize(self):
        bounds = [(-5.12, 5.12)] * 3
        ga = GeneticAlgorithm(pop_size=50, n_generations=20)
        results = ga.optimize(rastrigin_function, bounds)
        assert results["best_fitness"] > -50
        assert len(results["history"]) == 20
