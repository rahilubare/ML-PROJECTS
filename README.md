# Customer Churn Prediction using Random Forest

## **Project Overview**
This project predicts which customers are likely to **churn** (leave) a telecom company using machine learning.
It helps businesses **retain customers** by identifying at-risk users and taking proactive actions.

---

## **Features**
- Predicts customer churn with **Random Forest** (a powerful machine learning algorithm).
- Uses real-world telecom customer data.
- Easy to understand and deploy.

---

## **📌 Project Overview**
This project predicts which telecom customers are likely to **churn** (leave the company) using a **Random Forest** machine learning model.  
It helps businesses **retain customers** by identifying at-risk users and taking proactive actions.

---

## **📂 Project Structure**
customer_churn_prediction/
│
├── data/
│ ├── raw/
│ │ └── WA_Fn-UseC_-Telco-Customer-Churn.csv # Original dataset
│ └── processed/ # Cleaned and processed data
│
├── notebooks/
│ ├── 1_eda.ipynb # Exploratory Data Analysis
│ ├── 2_preprocessing.ipynb # Data cleaning and feature engineering
│ └── 3_modeling.ipynb # Model building and evaluation
│
├── src/
│ ├── preprocess.py # Script for data preprocessing
│ └── model.py # Script for model training and evaluation
│
├── models/
│ └── random_forest_model.pkl # Saved model
│
├── requirements.txt # Python dependencies
└── README.md # Project documentation

yaml
Copy code

---

## **🚀 How to Run This Project**

### **📋 Prerequisites**
- Python 3.8+
- Jupyter Notebook (for notebooks)
- Git (optional, for cloning)

---

### **🛠️ Step 1: Clone the Repository**
```bash
git clone https://github.com/yourusername/ML-Projects.git
cd ML-Projects/customer_churn_prediction
(Skip this step if you already have the project folder.)

📦 Step 2: Install Dependencies
bash
Copy code
pip install -r requirements.txt
Installs all required Python libraries.

🖥️ Step 3: Run the Project
Option 1: Using Jupyter Notebooks (Recommended for Beginners)
Start Jupyter Notebook:

bash
Copy code
jupyter notebook
Open and run the notebooks in order:

notebooks/1_eda.ipynb – Explore the data

notebooks/2_preprocessing.ipynb – Clean and split the data

notebooks/3_modeling.ipynb – Train and evaluate the model

Outputs:

Processed data → data/processed/

Trained model → models/random_forest_model.pkl

Option 2: Using Python Scripts (For Automation)
Preprocess the data:

bash
Copy code
python src/preprocess.py
Train and evaluate the model:

bash
Copy code
python src/model.py
Outputs:

Processed data → data/processed/

Trained model → models/random_forest_model.pkl

🔮 Step 4: Use the Trained Model
To make predictions with the trained model:

python
Copy code
import joblib
import pandas as pd

# Load the model
model = joblib.load('models/random_forest_model.pkl')

# Load new data (must be preprocessed the same way)
new_data = pd.read_csv('data/processed/X_test.csv')  # Example

# Predict
predictions = model.predict(new_data)
print(predictions)
📊 Data Description
Dataset: Telco Customer Churn from Kaggle

Features: Customer demographics, account info, services used, payment history

Target: Churn (Yes/No)

📈 Model Performance
Accuracy: ~85%

Key Features: Contract type, tenure, monthly charges

🤝 How to Contribute
Fork the repo

Make your changes

Submit a pull request
