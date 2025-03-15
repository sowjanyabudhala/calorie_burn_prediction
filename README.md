# 🔥 Calorie Burn Prediction using Gradient Boosting Regressor

## 📌 Table of Contents
- [Demo](#demo)
- [Overview](#overview)
- [Motivation](#motivation)
- [Technical Aspect](#technical-aspect)
- [Installation](#installation)
- [Run](#run)
- [Deployment (Streamlit)](#deployment-streamlit)
- [Directory Structure](#directory-structure)
- [To-Do](#to-do)
- [Bug / Feature Request](#bug--feature-request)
- [Technologies Used](#technologies-used)
- [Credits](#credits)

---

## 🎥 Demo
🔗 **Live Demo**: [Add your deployment link here]  

![Demo](https://your-demo-link.com/demo.gif)

---

## 📖 Overview
This project predicts **calories burned during exercise** using machine learning techniques, specifically the **Gradient Boosting Regressor**. The dataset includes information about **age, height, weight, gender, heart rate, body temperature, and exercise duration**.

💡 **Models Used**:
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor (**Best Performing Model**)

📊 **Evaluation Metrics**:
- R² Score

---

## 🎯 Motivation
Tracking calorie burn is crucial for **fitness and health**. This project aims to build an accurate **calorie burn prediction model** to help individuals monitor their workout effectiveness.  

---

## ⚙️ Technical Aspect
This project consists of **two main components**:
1. **Model Training & Evaluation**:
   - **Data Preprocessing**: Handling missing values, feature scaling, and encoding categorical data.
   - **Feature Engineering**: Creating relevant input features.
   - **Model Training**: Comparing Linear Regression, Random Forest, and Gradient Boosting.
   - **Model Evaluation**: Using R² Score for performance comparison.
   - **Best Model Selection**: Gradient Boosting Regressor is used for final predictions.

2. **Model Deployment using Streamlit**:
   - A **user-friendly UI** where users can input exercise data.
   - Predicts **calories burned** based on the input.

---

## 💻 Installation
Ensure that **Python 3.7 or above** is installed: [Download Python](https://www.python.org/downloads/).  

To install dependencies, run:
```bash
pip install -r requirements.txt

##🚀 Run
Step 1: Setting Environment Variables
python calorie_burn_prediction_by_gradient_boosting_regressor.py


🌐 Deployment (Streamlit)
To deploy this model using Streamlit, follow these steps:

Step 1: Install Streamlit
pip install streamlit

Step 2: Run the App
streamlit run app.py

📂 Directory Structure
📦 Calorie Burn Prediction
 ┣ 📂 data
 ┃ ┣ 📄 calories.csv
 ┃ ┣ 📄 exercise.csv
 ┣ 📂 models
 ┃ ┣ 📄 calorie_burn_prediction.pkl
 ┣ 📂 scripts
 ┃ ┣ 📄 calorie_burn_prediction_by_gradient_boosting_regressor.py
 ┣ 📂 streamlit_app
 ┃ ┣ 📄 app.py
 ┣ 📄 requirements.txt
 ┣ 📄 README.md
 ┣ 📄 LICENSE

✅ To-Do
✔ Deploy the model using Flask API
✔ Enable real-time calorie tracking

🐞 Bug / Feature Request
If you find a bug or want to request a new feature, open an issue:
📌 GitHub Issues

## 🛠 Technologies Used
### **1️⃣ Scikit-learn (`sklearn`)**
- **Terminology**: Machine Learning Library  
- **Description**: Provides tools for data preprocessing, classification, regression, clustering, and model evaluation.  
- **Usage**: Used for train-test splitting, model training, and evaluation metrics (R² Score).  
📌 [Scikit-learn Website](https://scikit-learn.org/)  

### **2️⃣ Streamlit (`streamlit`)**
- **Terminology**: Web App Framework for Machine Learning Models  
- **Description**: Enables real-time user interaction and model deployment through a web UI.  
- **Usage**: Used for deploying the calorie burn prediction model.  
📌 [Streamlit Website](https://streamlit.io/)  

### **3️⃣ Pandas (`pandas`)**
- **Terminology**: Data Analysis & Manipulation Library  
- **Description**: Helps in handling structured datasets (CSV, Excel) and performing data preprocessing.  
- **Usage**: Used for loading, cleaning, and transforming exercise and calorie data.  
📌 [Pandas Website](https://pandas.pydata.org/)  

### **4️⃣ Matplotlib (`matplotlib`)**
- **Terminology**: Data Visualization Library  
- **Description**: Used for generating graphs and charts to analyze trends in calorie burn.  
- **Usage**: Used for histograms, scatter plots, and heatmaps.  
📌 [Matplotlib Website](https://matplotlib.org/)  

🙌 Credits
Dataset: Kaggle Exercise & Calorie Burn Dataset
Contributors: B. Sowjanya


