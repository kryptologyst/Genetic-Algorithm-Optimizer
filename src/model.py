import numpy as np
from loguru import logger
from typing import Callable


class GeneticAlgorithm:
    def __init__(
        self,
        pop_size: int = 100,
        n_generations: int = 50,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        elitism: int = 2,
        random_state: int = 42,
    ):
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism = elitism
        self.rng = np.random.default_rng(random_state)
        self.history_: list = []

    def optimize(
        self, fitness_fn: Callable, bounds: list, chromosome_length: int = None,
    ) -> dict:
        if chromosome_length is None:
            chromosome_length = len(bounds)
        pop = self.rng.uniform(
            [b[0] for b in bounds], [b[1] for b in bounds],
            (self.pop_size, chromosome_length),
        )
        best_solution = None
        best_fitness = -np.inf
        for gen in range(self.n_generations):
            fitness = np.array([fitness_fn(ind) for ind in pop])
            elite_idx = np.argsort(fitness)[-self.elitism:]
            elites = pop[elite_idx].copy()
            if fitness[elite_idx[-1]] > best_fitness:
                best_fitness = fitness[elite_idx[-1]]
                best_solution = pop[elite_idx[-1]].copy()
            new_pop = elites.copy()
            while len(new_pop) < self.pop_size:
                p1 = self._tournament_select(pop, fitness)
                p2 = self._tournament_select(pop, fitness)
                if self.rng.random() < self.crossover_rate:
                    c1, c2 = self._crossover(p1, p2)
                else:
                    c1, c2 = p1.copy(), p2.copy()
                c1 = self._mutate(c1, bounds)
                c2 = self._mutate(c2, bounds)
                new_pop = np.vstack([new_pop, [c1]])
                if len(new_pop) < self.pop_size:
                    new_pop = np.vstack([new_pop, [c2]])
            pop = new_pop[:self.pop_size]
            self.history_.append(float(best_fitness))
            if (gen + 1) % 10 == 0:
                logger.info(f"Gen {gen+1}/{self.n_generations}: best={best_fitness:.4f}")
        logger.info(f"Final best fitness: {best_fitness:.4f}")
        return {
            "best_solution": best_solution.tolist(),
            "best_fitness": float(best_fitness),
            "history": self.history_,
        }

    def _tournament_select(self, pop, fitness, k=3):
        idx = self.rng.choice(len(pop), k)
        return pop[idx[np.argmax(fitness[idx])]]

    def _crossover(self, p1, p2):
        alpha = self.rng.random(len(p1))
        c1 = alpha * p1 + (1 - alpha) * p2
        c2 = alpha * p2 + (1 - alpha) * p1
        return c1, c2

    def _mutate(self, ind, bounds):
        for i in range(len(ind)):
            if self.rng.random() < self.mutation_rate:
                ind[i] += self.rng.normal(0, (bounds[i][1] - bounds[i][0]) * 0.1)
                ind[i] = np.clip(ind[i], bounds[i][0], bounds[i][1])
        return ind
