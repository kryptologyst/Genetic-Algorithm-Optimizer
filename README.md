# Genetic Algorithm Optimizer

**Evolutionary optimization** with tournament selection, blend crossover, and Gaussian mutation.

## Overview

- Optimizes Rastrigin and Sphere benchmark functions
- Tournament selection, blend crossover, elitism
- Convergence tracking across generations
- **Streamlit dashboard**

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
# CLI: python -m src.main optimize --problem rastrigin
pytest tests/ -v
```

## Docker

```bash
docker compose up --build
```

## License

MIT
# Genetic-Algorithm-Optimizer
