# Heart Disease Prediction

End-to-end machine learning project for estimating heart disease risk from clinical features. It includes exploratory analysis and model training in Jupyter notebooks, plus a **Streamlit** web app for interactive predictions.

> **Disclaimer:** This tool is for education and demonstration only. It is **not** medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional for health decisions.

## Features

- **Web app** (`src/app.py`): Enter patient-style inputs (age, vitals, ECG-related fields, chest pain type, etc.) and get a risk label with predicted probability.
- **Notebooks**: Data exploration (`notebooks/HeartAnalysis.ipynb`) and model work (`notebooks/heart_attack_models.ipynb`).
- **Data pipeline**: Raw data under `data/raw/`; processed exports under `data/processed/` (e.g. `heart_disease_processed.csv`).

## Tech Stack

- Python 3.10+ (recommended)
- `pandas`, `numpy`, `scikit-learn`, `joblib`
- `streamlit` for the UI
- `jupyter` for notebooks
- `matplotlib`, `seaborn` for visualization

## Project Structure

```
├── data/
│   ├── raw/                 # Original dataset (e.g. heart.csv)
│   └── processed/           # Cleaned / feature-ready data
├── models/                  # Trained artifacts (*.pkl) — see below
├── notebooks/               # EDA and training experiments
├── src/
│   └── app.py               # Streamlit application
├── requirements.txt
└── README.md
```

## Setup

1. **Clone or open** this repository.

2. **Create a virtual environment** (recommended):

   ```bash
   python -m venv .venv
   ```

   Activate it:

   - Windows (PowerShell): `.venv\Scripts\Activate.ps1`
   - macOS / Linux: `source .venv/bin/activate`

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Model Artifacts

The app expects these files in the **`models/`** directory (project root):

| File | Purpose |
|------|---------|
| `Logistic_Regression_heart.pkl` | Trained classifier |
| `scaler.pkl` | Fitted `StandardScaler` (or equivalent) |
| `columns.pkl` | Ordered list of feature column names after encoding |

These files are produced by the training notebook/workflow. They are listed in `.gitignore` by default, so you may need to **train the model locally** or copy artifacts into `models/` before running the app.

## Run the Web App

Run Streamlit **from the project root** so paths to `models/` resolve correctly:

```bash
streamlit run src/app.py
```

Then open the URL shown in the terminal (usually `http://localhost:8501`).

## Notebooks

- Start Jupyter from the project root (or open files in your IDE):

  ```bash
  jupyter notebook
  ```

- Use `notebooks/HeartAnalysis.ipynb` for exploration and `notebooks/heart_attack_models.ipynb` for model comparison and exporting the `.pkl` files used by the app.

## Input Fields (App)

The UI collects features aligned with the dataset, including: age, sex, resting blood pressure, cholesterol, fasting blood sugar flag, max heart rate, oldpeak, exercise angina, chest pain type, resting ECG, and ST slope. Categorical fields are one-hot encoded to match `columns.pkl` before scaling and prediction.

## License

Use and modify for learning and portfolio purposes. Add a license file if you plan to distribute the project publicly.
