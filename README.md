# Customer_Lifetime_Value_Prediction

> A machine learning pipeline that predicts how much a customer is worth to your business — using Linear Regression and Random Forest, with a production-ready FastAPI deployment.

---

##  Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Models & Evaluation](#models--evaluation)
- [Getting Started](#getting-started)
- [Deployment](#deployment)
- [Requirements](#requirements)
- [License](#license)

---

##  Overview

Customer Lifetime Value (CLV) is a key business metric that estimates the total revenue a customer will generate over their relationship with your company. This project builds, evaluates, and deploys a CLV prediction model using two approaches:

- **Linear Regression** — a fast, interpretable baseline
- **Random Forest Regressor** — a powerful ensemble model that handles complex, non-linear patterns

The trained model is saved with `joblib` and served via a **FastAPI** REST API for real-time predictions.

---

##  Project Structure

```
customer-lifetime-value/
│
├── Customer_lifetime.ipynb     # Main notebook: EDA, training & evaluation
├── customer_lifetime.csv       # Raw dataset
├── CLV_model.joblib            # Saved trained model
├── modelfeatures.joblib        # Saved feature schema
├── app.py                      # FastAPI app for serving predictions
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

##  Dataset

The dataset (`customer_lifetime.csv`) contains customer-level features used to predict CLV. The target variable is:

| Column | Description |
|--------|-------------|
| `CLV`  | Customer Lifetime Value *(target)* |
| Other columns | Customer features used as model inputs |

---

##  Models & Evaluation

### Models Trained

| Model | Key Parameters |
|-------|---------------|
| Linear Regression | scikit-learn defaults |
| Random Forest Regressor | `n_estimators=200`, `random_state=42` |

### Evaluation Metrics

| Metric | What It Measures |
|--------|-----------------|
| **RMSE** | Average prediction error in CLV units — lower is better |
| **R² Score** | How much variance the model explains — closer to 1.0 is better |

```python
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

RMSE_linear = sqrt(mean_squared_error(y_test, Predictions))
r2_linear   = r2_score(y_test, Predictions)

RMSE_tree   = sqrt(mean_squared_error(y_test, random_prediction))
r2_tree     = r2_score(y_test, random_prediction)
```

---

##  Getting Started

### Prerequisites

- Python 
- pip
- Jupyter Notebook

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/customer-lifetime-value.git
cd customer-lifetime-value
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch the Notebook

```bash
jupyter notebook Customer_lifetime.ipynb
```

Run all cells to load the data, train both models, evaluate performance, and save the model files.

---

##  Deployment

### Option 1 — FastAPI (Local)

The trained model is served via a lightweight REST API.

```bash
uvicorn app:app --reload
```

| URL | Description |
|-----|-------------|
| `http://localhost:8000/predict` | Prediction endpoint |
| `http://localhost:8000/docs` | Auto-generated interactive API docs |

**Example Request:**

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"feature_1": 3.5, "feature_2": 1200.0}'
```

**Example Response:**

```json
{
  "predicted_clv": 4821.35
}
```

---


##  Requirements

```
pandas
scikit-learn
matplotlib
joblib
fastapi
uvicorn
jupyter
```

Install all at once:

```bash
pip install -r requirements.txt
```

---

##  Workflow Summary

```
Load CSV  ──►  Explore & Clean  ──►  Train/Test Split (80/20)
                                             │
                              ┌──────────────┴─────────────┐
                              ▼                             ▼
                    Linear Regression           Random Forest Regressor
                              │                             │
                              └──────────────┬─────────────┘
                                             ▼
                                  Evaluate (RMSE, R²)
                                             │
                                             ▼
                                   Save Model (joblib)
                                             │
                                             ▼
                                  Deploy via FastAPI
```

---


##  License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">
  Made with  using Python, scikit-learn & FastAPI
</div>
