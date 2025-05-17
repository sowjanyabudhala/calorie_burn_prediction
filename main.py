import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import metrics

# Title of the Web App
st.title("Calorie Burn Prediction App")

# Load datasets
@st.cache_data
def load_data():
    calories = pd.read_csv('calories.csv')
    exercise = pd.read_csv('exercise.csv')
    df = pd.merge(calories, exercise, on="User_ID")
    df.replace({'Gender': {'male': 0, 'female': 1}}, inplace=True)
    return df

df = load_data()

# Show dataset
if st.checkbox("Show Dataset"):
    st.write(df.head())

# Splitting features and target
X = df.drop(['Calories', 'User_ID'], axis=1)
y = df['Calories']

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Model Training
gbr = GradientBoostingRegressor()
gbr.fit(X_train, y_train)

# Model Evaluation
y_pred = gbr.predict(X_test)
r2_score = metrics.r2_score(y_test, y_pred)
st.write(f"Model R² Score: {r2_score:.2f}")

# Sidebar Input Form
st.sidebar.header("Enter User Details")
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
age = st.sidebar.number_input("Age", min_value=1, max_value=100)
height = st.sidebar.number_input("Height (cm)")
weight = st.sidebar.number_input("Weight (kg)")
duration = st.sidebar.number_input("Exercise Duration (min)")
heart_rate = st.sidebar.number_input("Heart Rate (bpm)")
body_temp = st.sidebar.number_input("Body Temperature (°C)")

# Convert gender to numeric
gender = 0 if gender == "Male" else 1

# Prediction button
if st.sidebar.button("Predict Calories Burned"):
    user_data = pd.DataFrame([[gender, age, height, weight, duration, heart_rate, body_temp]],
                             columns=['Gender', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp'])
    prediction = gbr.predict(user_data)
    st.sidebar.success(f"Estimated Calories Burned: {prediction[0]:.2f} kcal")

# Correlation Heatmap
st.subheader("Feature Correlation Heatmap")
plt.figure(figsize=(6,4))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
st.pyplot(plt)
