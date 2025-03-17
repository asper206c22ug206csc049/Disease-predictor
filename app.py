import pandas as pd
import numpy as np
import sqlite3
import os
from flask import Flask, render_template, request

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)

# Load data from CSV files
def load_data():
    try:
        train_data = pd.read_csv(r"D:\Desktop\diseasepredictor\database\training.csv")
        test_data = pd.read_csv(r"D:\Desktop\diseasepredictor\database\testing.csv")
        return train_data, test_data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

# Preprocess data
def preprocess_data(train_data, test_data):
    try:
        diseases = train_data['prognosis'].unique()
        disease_to_num = {disease: i for i, disease in enumerate(diseases)}
        
        # Replace disease names with numerical labels
        train_data['prognosis'] = train_data['prognosis'].replace(disease_to_num).astype(int)
        test_data['prognosis'] = test_data['prognosis'].replace(disease_to_num).astype(int)

        X_train = train_data.iloc[:, :-1]
        y_train = train_data['prognosis']
        X_test = test_data.iloc[:, :-1]
        y_test = test_data['prognosis']

        return X_train, y_train, X_test, y_test, diseases
    except Exception as e:
        print(f"Error preprocessing data: {e}")
        return None, None, None, None, None

# Save predictions to the database
def save_to_db(name, symptoms, disease):
    try:
        os.makedirs("database", exist_ok=True)
        conn = sqlite3.connect('database/database.db')
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                symptom1 TEXT,
                symptom2 TEXT,
                symptom3 TEXT,
                symptom4 TEXT,
                symptom5 TEXT,
                disease TEXT
            )
        """)
        c.execute("""
            INSERT INTO predictions (name, symptom1, symptom2, symptom3, symptom4, symptom5, disease)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (name, *symptoms, disease))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error saving to database: {e}")

# Load and preprocess data
train_data, test_data = load_data()
if train_data is not None and test_data is not None:
    X_train, y_train, X_test, y_test, diseases = preprocess_data(train_data, test_data)
else:
    X_train, y_train, X_test, y_test, diseases = None, None, None, None, None

# Train models
if X_train is not None and y_train is not None:
    decision_tree_model = DecisionTreeClassifier().fit(X_train, y_train)
    random_forest_model = RandomForestClassifier(n_estimators=100).fit(X_train, y_train)
    naive_bayes_model = GaussianNB().fit(X_train, y_train)
    knn_model = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
else:
    decision_tree_model = random_forest_model = naive_bayes_model = knn_model = None

# Prediction function
def predict_disease(input_symptoms, model, diseases):
    try:
        predicted_disease = diseases[model.predict([input_symptoms])[0]]
        return predicted_disease
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None

@app.route('/')
def home():
    return render_template('index.html', symptoms=X_train.columns if X_train is not None else [])

@app.route('/predict', methods=['POST'])
def predict():
    try:
        name = request.form.get('name', '').strip()
        symptoms = [request.form.get(f'symptom{i}', '').strip() for i in range(1, 6)]

        # Validate all fields
        if not name:
            return render_template('index.html', error="Name is required.", symptoms=X_train.columns if X_train is not None else [])
        if not all(symptoms):
            return render_template('index.html', error="All symptoms must be selected.", symptoms=X_train.columns if X_train is not None else [])

        # Convert symptoms to binary input
        input_symptoms = [1 if symptom in symptoms else 0 for symptom in X_train.columns]

        # Get predictions from all models
        predictions = {
            'Decision Tree': predict_disease(input_symptoms, decision_tree_model, diseases),
            'Random Forest': predict_disease(input_symptoms, random_forest_model, diseases),
            'Naive Bayes': predict_disease(input_symptoms, naive_bayes_model, diseases),
            'K-Nearest Neighbors': predict_disease(input_symptoms, knn_model, diseases)
        }

        # Save all predictions to database
        for model_name, prediction in predictions.items():
            if prediction:
                save_to_db(name, symptoms, prediction)

        return render_template('index.html', predictions=predictions, symptoms=X_train.columns if X_train is not None else [])
    except Exception as e:
        return render_template('index.html', error=f"An error occurred: {str(e)}", symptoms=X_train.columns if X_train is not None else [])

if __name__ == '__main__':
    app.run(debug=True)