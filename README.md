Mock Multi-Agent Business Model

Overview
--------
This repository contains a mock multi-agent system for a bird image classification and business-model proof-of-concept. The project demonstrates how lightweight agent chains and simple ML models can be composed to perform tasks like dataset construction, image analysis, calibration, and estimation.

Key Features
------------
- **Multi-agent flow:** Chains implemented under the `chains/` folder to coordinate tasks.
- **Image classification:** Includes an image analysis agent to use a .pt file for image recognition (trained_bird_classifier.pt).
- **Flask API:** Includes a simple Flask API to expose the workflow as a web service.

Repository Structure
--------------------
- **main.py**: Primary entrypoint for running the project (experiment runner / demo).
- **chains/**: Agent implementations and orchestrator.
- **graphs/**: Graph definitions and main_graph utilities.
- **templates/**: HTML templates for the Flask web interface.
- **static/**: Static assets (CSS, JS) for the web interface.

Requirements
------------
- Python 3.10+ recommended
- A virtual environment (recommended)
- Dependencies are declared in `pyproject.toml`; install via pip or your preferred tool.
- The trained model file `trained_bird_classifier.pt` is required for image classification tasks.
- An openai API key is required for agent logic. Put the API key either in OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL respectively or update the files in chain manually to your api information.
- Tavily Search API is needed for the knowledge search agent, put the API key in an environment variable named TAVILY_API_KEY.

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
open http://127.0.0.1:5000 on web browser of choice
```

- To run tests:

```powershell
pytest -q
```

Development Notes
-----------------
- Agent code lives in `chains/`. Each agent focuses on a single responsibility (calibration, construction, estimation, image analysis, orchestration).