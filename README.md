# UCLA Admission Chance Predictor

A production-quality machine learning web application that predicts the likelihood
of a student being admitted to UCLA's graduate program using a **Multi-Layer
Perceptron (MLP) Neural Network**.  The model is served through an interactive
**Streamlit** dashboard with three pages: Dataset Overview, Model Performance,
and live Admission Prediction.

---

## Project Description

Gaining admission to a top graduate programme is highly competitive.  This project
trains a binary classification neural network on 500 historical UCLA applicant
records.  The target variable (`Admit_Chance`) is binarised at the 0.80 threshold:
a student is labelled **admitted (1)** if their raw admission probability is >= 80%,
otherwise **not admitted (0)**.

The Streamlit front-end lets users explore the dataset, review full model
performance metrics (accuracy, confusion matrix, training loss curve, and
classification report), and enter their own academic profile to receive an instant
admission prediction with class probabilities.

---

## Project Structure

```
ucla_admission_predictor/
|
+-- app.py                  # Streamlit application entry point
+-- requirements.txt        # Python package dependencies
+-- README.md               # Project documentation (this file)
|
+-- data/
|   +-- Admission.csv       # UCLA admissions dataset (500 records)
|
+-- src/
    +-- __init__.py         # Source package initialiser
    +-- data_loader.py      # CSV loading and column validation
    +-- preprocessor.py     # Feature engineering, train/test split, MinMax scaling
    +-- model.py            # MLPClassifier training
    +-- evaluator.py        # Accuracy, confusion matrix, classification report
```

---

## Dataset

| Property      | Details                                       |
|---------------|-----------------------------------------------|
| Source        | UCLA Admissions dataset                       |
| File          | `data/Admission.csv`                          |
| Samples       | 500 applicant records                         |
| Target column | `Admit_Chance` (binarised: 1 if >= 0.8 else 0)|

### Feature Columns

| Column              | Type         | Description                                     |
|---------------------|--------------|-------------------------------------------------|
| `Serial_No`         | int          | Record identifier – dropped before training     |
| `GRE_Score`         | int          | GRE exam score (260–340)                        |
| `TOEFL_Score`       | int          | TOEFL exam score (92–120)                       |
| `University_Rating` | int (1–5)    | Undergraduate university prestige rating        |
| `SOP`               | float (1–5)  | Strength of Statement of Purpose                |
| `LOR`               | float (1–5)  | Strength of Letter of Recommendation            |
| `CGPA`              | float (1–10) | Cumulative undergraduate GPA                    |
| `Research`          | int (0/1)    | Research experience flag                        |
| `Admit_Chance`      | float (0–1)  | Raw admission probability (target variable)     |

---

## Model: MLP Neural Network

| Component           | Configuration                                   |
|---------------------|-------------------------------------------------|
| Algorithm           | `MLPClassifier` (scikit-learn)                  |
| Hidden layer sizes  | `(3,)` – one hidden layer with 3 neurons        |
| Activation function | `tanh`                                          |
| Optimiser           | Adam (scikit-learn default)                     |
| Batch size          | 50                                              |
| Max iterations      | 200                                             |
| Random state        | 123                                             |
| Scaler              | `MinMaxScaler` (fit on train, applied to both)  |
| Train / test split  | 80 % / 20 %, stratified, `random_state=123`     |
| Typical accuracy    | 90 %+ on training set, ~85–92 % on test set     |

### Preprocessing Pipeline

1. Drop `Serial_No` identifier column.
2. Binarise `Admit_Chance`: **1** if >= 0.80, else **0**.
3. Cast `University_Rating` and `Research` to `object` dtype.
4. One-hot encode `University_Rating` (5 levels) and `Research` (2 levels)
   via `pd.get_dummies(..., dtype=int)`.
5. `train_test_split` with `test_size=0.2`, `random_state=123`, `stratify=y`.
6. Fit `MinMaxScaler` on training features; transform both train and test sets.

---

## Setup Instructions

### 1. Clone / download the project

```bash
git clone <repository-url>
cd ucla_admission_predictor
```

### 2. (Recommended) Create a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Confirm the dataset is in place

```
ucla_admission_predictor/data/Admission.csv
```

### 5. Launch the Streamlit application

Run the following command from inside the `ucla_admission_predictor/` directory:

```bash
streamlit run app.py
```

The app opens automatically in your default browser at `http://localhost:8501`.

---

## Application Pages

### Dataset Overview
- Summary metrics: total students, admission rate, average CGPA.
- Raw data table (first 10 rows) and full statistical summary.
- Correlation heatmap for all numeric features.
- Histogram distributions for GRE Score, TOEFL Score, and CGPA.

### Model Performance
- Train and test accuracy displayed as Streamlit metrics.
- Seaborn confusion matrix heatmap for the test set.
- Training loss curve plotted over all iterations.
- Full classification report (precision, recall, F1-score per class).
- Model architecture summary card.

### Predict Admission
- Interactive sliders and dropdowns for all seven input features.
- One-click prediction button that returns:
  - Admitted / Not Admitted label.
  - Admission probability and rejection probability.

---

## Dependencies

| Package        | Purpose                                        |
|----------------|------------------------------------------------|
| `streamlit`    | Interactive web application framework          |
| `pandas`       | Data loading and manipulation                  |
| `numpy`        | Numerical operations                           |
| `scikit-learn` | MLPClassifier, MinMaxScaler, metrics           |
| `matplotlib`   | Loss curve and distribution plots              |
| `seaborn`      | Confusion matrix heatmap                       |

---

## Notes

- The trained model and scaler are cached with `@st.cache_resource`, so they
  are only recomputed on a fresh app start or explicit cache clear.
- Prediction inputs are manually one-hot encoded to exactly match the
  `feature_columns` order produced during training, ensuring correct alignment
  with the fitted scaler.
- All source modules use Python's `logging` module for structured log output
  and `try/except` blocks for robust error handling.
- The dataset uses `Admission.csv` (CSV format).  If only the Excel version
  (`Admission.xls`) is available, convert it to CSV first or change the
  `DATA_PATH` constant in `app.py` and switch `pd.read_csv` to
  `pd.read_excel` in `src/data_loader.py`.
