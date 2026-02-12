# Project: Wingspan Card Analytics

## Workflow
- Use traditional git branching from the main repo directory: `C:\Users\User\Projects\wingspan_cards`
- Do NOT use git worktrees. Create feature branches, merge to main.
- Python virtual environment lives in `venv/` (gitignored).

## Tech Stack
- Python 3, pip + venv
- pandas, numpy, scipy for analysis
- matplotlib, seaborn for visualization
- jupyter for notebooks

## Project Structure
- `assets/` — source data (master_bird_data.json, 596 bird cards)
- `src/` — Python modules (load_data.py provides `load_bird_data()`)
- `notebooks/` — Jupyter notebooks for exploratory analysis
- `output/` — generated files, plots (gitignored)

## Conventions
- Activate venv before running: `venv\Scripts\activate`
- Data loading: `from src.load_data import load_bird_data`
