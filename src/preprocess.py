import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def preprocess_data(input_path, output_dir):
    # Load data
    df = pd.read_csv(input_path)

    # Drop customerID
    df = df.drop('customerID', axis=1)

    # Encode target
    df['Churn'] = LabelEncoder().fit_transform(df['Churn'])

    # Encode categorical features
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col])

    # Handle missing values
    df = df.fillna(df.mean())

    # Split data
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Save processed data
    os.makedirs(output_dir, exist_ok=True)
    X_train.to_csv(f'{output_dir}/X_train.csv', index=False)
    X_test.to_csv(f'{output_dir}/X_test.csv', index=False)
    y_train.to_csv(f'{output_dir}/y_train.csv', index=False)
    y_test.to_csv(f'{output_dir}/y_test.csv', index=False)
    print(f"Processed data saved to {output_dir}")

if __name__ == "__main__":
    # Use absolute path for input and output
    input_path = 'C:/Project/customer_churn_prediction/data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv'
    output_dir = 'C:/Project/customer_churn_prediction/data/processed'
    preprocess_data(input_path, output_dir)
