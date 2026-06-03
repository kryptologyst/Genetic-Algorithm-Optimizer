import streamlit as st
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data import get_optimization_problem
from src.model import GeneticAlgorithm

st.set_page_config(page_title="Genetic Algorithm", page_icon="🧬", layout="wide")
st.title("🧬 Genetic Algorithm Optimizer")
st.markdown("Evolutionary optimization with tournament selection, blend crossover, and Gaussian mutation.")

problem = st.selectbox("Problem", ["rastrigin", "sphere"])
fn, bounds = get_optimization_problem(problem)

c1, c2, c3 = st.columns(3)
with c1:
    generations = st.slider("Generations", 10, 200, 50)
with c2:
    pop_size = st.slider("Population", 20, 500, 100, 20)
with c3:
    mutation_rate = st.slider("Mutation Rate", 0.01, 0.5, 0.1, 0.01)

if st.button("Run GA", type="primary"):
    with st.spinner(f"Evolving for {generations} generations..."):
        ga = GeneticAlgorithm(pop_size=pop_size, n_generations=generations, mutation_rate=mutation_rate)
        results = ga.optimize(fn, bounds)
    st.success(f"Best Fitness: **{results['best_fitness']:.4f}**")
    st.write("Best Solution:", {f"x{i}": f"{v:.4f}" for i, v in enumerate(results["best_solution"])})
    df = pd.DataFrame({"Generation": range(len(results["history"])), "Fitness": results["history"]})
    st.line_chart(df.set_index("Generation"))
