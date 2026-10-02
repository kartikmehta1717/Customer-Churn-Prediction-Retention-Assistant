# Customer Churn Prediction & Retention Assistant

GitHub repository: [kartikmehta1717/Customer-Churn-Prediction-Retention-Assistant](https://github.com/kartikmehta1717/Customer-Churn-Prediction-Retention-Assistant)

A Streamlit application that estimates customer churn risk, explains predictions with SHAP, retrieves relevant retention policies, and can draft suggestions with Gemini. The repository also includes Jupyter notebooks for the analysis and model workflow, plus optional scripts for exporting predictions to MySQL.

## Requirements

- Python 3.10 or newer (Python 3.10 is the version indicated by the checked-in bytecode; other versions may require compatible package versions).
- Git, to clone and version the project.
- The raw Telco customer CSV described below.
- A Gemini API key for LLM-generated suggestions. The app's local policy-based suggestions do not require the key.
- A running MySQL server only if you intend to use the database scripts.

## Setup

Clone the repository and open its root directory:

```bash
git clone https://github.com/kartikmehta1717/Customer-Churn-Prediction-Retention-Assistant.git
cd Customer-Churn-Prediction-Retention-Assistant
```

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in the values you need. Never commit `.env` or put real credentials in source files.

## Data and model files

Large datasets and generated model artifacts are intentionally excluded from Git. Put the source dataset at `data/raw/telco.csv`; the application and notebooks expect that file. The source CSV should contain the fields used in the notebooks, including `Customer ID` and `Churn Label`.

The application also expects generated files under `data/processed/` and `models/` (including the trained model, held-out test data, SHAP artifacts, and FAISS policy index). Recreate them by running the notebooks in order:

1. `notebooks/01_eda.ipynb`
2. `notebooks/02_model_training.ipynb`
3. `notebooks/03_shap_analysis.ipynb`
4. `notebooks/04_rag_setup.ipynb`
5. `notebooks/05_llm_integration.ipynb`

Run the notebooks from the repository root so their relative paths resolve. The `knowledge_base/` policy text files are tracked in the repository and are used by the RAG workflow.

## Run the app

After the data and artifacts are in place, from the repository root:

```bash
streamlit run app/app.py
```

The app loads model/data artifacts when it starts. If those files are missing, complete the notebook workflow first. Set `GEMINI_API_KEY` in `.env` to enable Gemini-generated responses; without it, use the local policy-based recommendations.

## Optional: MySQL scripts

Set `DB_USER`, `DB_PASSWORD`, `DB_HOST`, and `DB_NAME` in `.env`, create the database on your MySQL server, and then run the scripts from the repository root as needed:

```bash
python load_customers.py
python create_predictions.py
python load_predictions.py
python test_db.py
```

The data loading scripts replace their target tables when run. `create_predictions.py` requires the generated model, test data, and raw source CSV.

## Repository notes

- `.env`, Python caches, generated model artifacts, and raw/processed CSV files are ignored by Git.
- Do not commit customer data, API keys, database passwords, or other private credentials.
- `requirements.txt` lists the Python packages used by the app, notebooks, and optional database scripts.
