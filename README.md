# Student Marks Prediction System 

An end-to-end Machine Learning project that predicts student marks based on
study hours, attendance, and sleep hours. The trained model is saved and reused
for future predictions.

---

# Features
- Trains a Linear Regression model
- Uses multiple features for prediction
- Evaluates model using RMSE
- Saves trained model using Joblib
- Loads model for reuse (no retraining needed)

---

# Tech Stack
- Python
- Pandas
- Scikit-learn
- NumPy
- Joblib

---

# Project Structure
student-marks-prediction-ml/
│── students.csv
│── student_ml.py
│── student_marks_model.pkl
│── README.md


---

# Model Evaluation
- Metric used: **RMSE (Root Mean Squared Error)**
- Lower RMSE indicates better model performance

---

# How to Run
```bash
pip install pandas scikit-learn numpy joblib
python student_ml.py
