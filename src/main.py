import typer
import sys
from loguru import logger

from .config import settings
from .data import get_optimization_problem
from .model import GeneticAlgorithm
from .visualizer import GAVisualizer

app = typer.Typer(help="Genetic Algorithm Optimizer CLI")
logger.remove()
logger.add(sys.stderr, level=settings.log_level)


@app.command()
def optimize(
    problem: str = typer.Option("rastrigin", help="Problem: rastrigin, sphere"),
    generations: int = typer.Option(50, help="Number of generations"),
    pop_size: int = typer.Option(100, help="Population size"),
):
    logger.info(f"Optimizing {problem} with GA ({generations} gens, {pop_size} pop)...")
    fn, bounds = get_optimization_problem(problem)
    ga = GeneticAlgorithm(pop_size=pop_size, n_generations=generations)
    results = ga.optimize(fn, bounds)
    logger.info(f"Best fitness: {results['best_fitness']:.4f}")
    logger.info(f"Best solution: {[f'{x:.3f}' for x in results['best_solution']]}")
    GAVisualizer.plot_convergence(results["history"], save_path=settings.plots_dir / "ga_convergence.png")
    logger.success("Done!")


if __name__ == "__main__":
    app()
