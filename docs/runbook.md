# Runbook

## Python backend

Directory: `src/backend/python_app/`

- Install deps: `pip install -r requirements.txt`
- Start app: `python app.py`
- Alternate start: `python app/run.py`
- Root helper script (PowerShell): `.\scripts\run-python.ps1`

## Node backend

Directory: `src/backend/node_api/`

- Install deps: `npm install`
- Start app: `npm start`
- Dev mode: `npm run dev`
- Root helper scripts (PowerShell):
  - `.\scripts\run-node.ps1`
  - `.\scripts\dev-node.ps1`

## Notebooks

Directory: `notebooks/`

- Read raw data from `../data/raw/`
- Write generated outputs to `../data/processed/`
- Scratch notebook renamed to `scratch_experiment.ipynb`
