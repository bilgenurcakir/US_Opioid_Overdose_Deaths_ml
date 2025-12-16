# US Opioid Overdose Death Rate Prediction

This project analyzes opioid overdose data in the United States and builds
machine learning models to predict the **Crude Overdose Death Rate** based on
demographic, temporal, and prescription-related features.

---

## 📊 Dataset

**Source:**  
 kaggle- US Opioid Overdose Deaths-[https://www.kaggle.com/datasets/thedevastator/us-opioid-overdose-deaths]
**Features used:**
- Year
- Population
- Deaths
- Prescription volume
- State (One-Hot Encoded)

**Target Variable:**
- Crude Rate (Deaths per 100,000 population)

---

## 🧠 Models Used

The following regression models were trained and compared:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- Support Vector Regression (SVR)
- K-Nearest Neighbors (KNN)

---

## 📈 Model Evaluation

Models were evaluated using:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score
- Accuracy (%) = R² × 100

### Best Performing Model:
**Random Forest Regressor**

---

## 🔍 Feature Importance

For tree-based models, feature importance was extracted.
Top contributing features include:
- Population
- Deaths
- Year
- Certain states (e.g., West Virginia)

---

## 📉 Visualizations

- Actual vs Predicted Crude Rate
- Feature importance bar chart

---

## 🚀 How to Run

```bash
git clone https://github.com/USERNAME/US_Opioid_Overdose_Deaths_ml.git
cd US_Opioid_Overdose_Deaths_ml
pip install -r requirements.txt
python main.py
