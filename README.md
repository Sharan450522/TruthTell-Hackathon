# TruthTell Hackathon

Clean Python-first repository layout for backend services, notebooks, and data workflows.

## Project layout

- `src/backend/python_app/` Python Flask backend
- `src/backend/node_api/` Node backend
- `notebooks/` model and experiment notebooks
- `data/raw/` local source datasets
- `data/processed/` generated outputs
- `artifacts/` runtime/model artifacts
- `archive/zips/` archived zip bundles
- `docs/` setup and run instructions

## Run

Use `docs/runbook.md` for setup and start commands.
Use `docs/model_workflow_and_methods.md` for datasets, fine-tuning workflow, and methods summary.

Quick launch from project root (PowerShell):

- `.\scripts\run-python.ps1`
- `.\scripts\run-node.ps1`
- `.\scripts\dev-node.ps1`
