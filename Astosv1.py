# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib  # For model saving/loading

# Step 1: Data Loading
def load_data(file_path):
    return pd.read_csv(file_path, delimiter='\t' if file_path.endswith('.tsv') else ',')

# Step 2: Data Preprocessing
def preprocess_data(df):
    # Example preprocessing steps
    df.fillna(0, inplace=True)
    # Assuming 'tools_config' and 'vulnerability_severity' columns exist
    X = df['tools_config']
    y = df['vulnerability_severity']
    
    # Ensure y has consistent data types
    y = np.array(y)
    try:
        y = y.astype(int)
    except ValueError:
        y = y.astype(str)
    
    return X, y

# Step 3: Feature Engineering
def feature_engineering(X):
    # Convert configurations into features (example, use vectorization)
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer()
    X_vectorized = vectorizer.fit_transform(X)
    return X_vectorized

# Step 4: Model Training
def train_model(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

# Step 5: Model Evaluation
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.show()

# Step 6: Visualize Results
def visualize_results(df):
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='vulnerability_severity')
    plt.title('Vulnerability Severity Distribution')
    plt.show()

# Example usage
file_path = 'dataset.csv'  # Ensure this path is correct
data = load_data(file_path)
X, y = preprocess_data(data)
X_vectorized = feature_engineering(X)
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.3, random_state=42)

model = train_model(X_train, y_train)
evaluate_model(model, X_test, y_test)
visualize_results(data)

# Save the model
joblib.dump(model, 'threat_model.pkl')
