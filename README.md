Mock Multi-Agent Business Model

Overview
--------
This repository contains a mock multi-agent system for a bird image classification and business-model proof-of-concept. The project demonstrates how lightweight agent chains and simple ML models can be composed to perform tasks like dataset construction, image analysis, calibration, and estimation.

Key Features
------------
- **Multi-agent flow:** Chains implemented under the `chains/` folder to coordinate tasks.
- **Image classification:** Includes an image analysis agent to use a .pt file for image recognition (not included in this repository due to size.).

Repository Structure
--------------------
- **main.py**: Primary entrypoint for running the project (experiment runner / demo).
- **chains/**: Agent implementations and orchestrator.
- **graphs/**: Graph definitions and main_graph utilities.

Requirements
------------
- Python 3.10+ recommended
- A virtual environment (recommended)
- Dependencies are declared in `pyproject.toml`; install via pip or your preferred tool.

Quick Setup
-----------
1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies (using pip):

```powershell
pip install -U pip
pip install -e .  # or: pip install -r requirements.txt if you have one
```

Running the Project
-------------------
- Run the main demo / experiment runner:

```powershell
python main.py
```

- To run tests:

```powershell
pytest -q
```

Development Notes
-----------------
- Agent code lives in `chains/`. Each agent focuses on a single responsibility (calibration, construction, estimation, image analysis, orchestration).