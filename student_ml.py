import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
import numpy as np

# Load data
df = pd.read_csv("students.csv")

# Add extra features
df['attendance'] = [70, 75, 80, 85, 90, 88]
df['sleep_hours'] = [6, 7, 8, 6, 7, 8]

# Features & target
X = df[['study_hours', 'attendance', 'sleep_hours']]
y = df['marks']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print("RMSE:", rmse)

# Save model
joblib.dump(model, "student_marks_model.pkl")

# Test prediction
sample = pd.DataFrame(
    [[7, 85, 7]],
    columns=['study_hours', 'attendance', 'sleep_hours']
)

print("Predicted marks:", model.predict(sample))
