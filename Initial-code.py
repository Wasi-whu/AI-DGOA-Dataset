import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import make_classification

# Load dataset from uploaded CSV file

# Upload the CSV file

# Assuming the uploaded file is named 'datasets-AI-DGOA.csv'
df = pd.read_csv('datasets-AI-DGOA.csv')

# Extract features and target variable from the DataFrame
# Adjust the following line according to the actual target variable in your dataset
X = df.iloc[:, :-4]  # Features (T1, T2, T3, T4, X1, X2, X3, X4)
y = df.iloc[:, -4]   # Target variable (Y1, Y2, Y3, Y4) - adjust as necessary

# Parameters
num_observations = 10000  # Initial number of observations
symmetry_factor = 3  # Symmetry factor
expanded_observations = num_observations * (symmetry_factor + 1)  # 10,000 * 6
num_predictive_features = 12
num_nodes = 4  # Total nodes (3 consumer + 1 central node)
episodes = 100  # Number of episodes

# Generate synthetic dataset
def generate_synthetic_data(num_samples, num_features):
    X, y = make_classification(n_samples=num_samples, n_features=num_features, n_informative=8, n_redundant=2, random_state=42)
    return X, y

# Create the dataset
print("Generating synthetic dataset...")
X, y = generate_synthetic_data(expanded_observations, num_predictive_features)
print(f"Dataset created with {expanded_observations} observations and {num_predictive_features} features.")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Dataset split into training and testing sets.")

# Initialize Random Forest model with GridSearchCV for hyperparameter tuning
rf_model = RandomForestClassifier(random_state=42)

# Hyperparameter tuning for Random Forest
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)
best_rf_model = grid_search.best_estimator_

print("Best Random Forest parameters found:", grid_search.best_params_)

# Initialize other models
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
voting_model = VotingClassifier(estimators=[
    ('rf', best_rf_model),
    ('gb', gb_model)
], voting='soft')
stacked_model = StackingClassifier(estimators=[
    ('rf', best_rf_model),
    ('gb', gb_model)
], final_estimator=LogisticRegression())

# Store performance metrics
metrics = []

# Train models for the specified number of episodes
print("Starting training of models...")
for episode in range(1, episodes + 1):
    best_rf_model.fit(X_train, y_train)
    gb_model.fit(X_train, y_train)
    voting_model.fit(X_train, y_train)
    stacked_model.fit(X_train, y_train)
    
    if episode % 10 == 0 or episode == 1:
        print(f"Episode {episode}/{episodes} completed.")

# Predictions and evaluation
print("Evaluating models...")
models = {
    "m1": m1_rf_model,
    "m2": m2_model,
    "m3": m3_model,
    "m4": m4_model
}

# Evaluate each model and store metrics
for model_name, model in models.items():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    metrics.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": report['1']['precision'],
        "Recall": report['1']['recall'],
        "F1-score": report['1']['f1-score'],
        "Support": report['1']['support']
    })
    
    print(f"{model_name} - Accuracy: {accuracy:.4f}, Precision: {report['1']['precision']:.4f}, "
          f"Recall: {report['1']['recall']:.4f}, F1-score: {report['1']['f1-score']:.4f}")

# Specify the path to save the CSV file
save_path = r'C:\code\project1\venv\2model_performance_metrics.csv'  # Use raw string to handle backslashes

# Create a DataFrame and save to CSV
metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv(save_path, index=False)

print(f"Model performance metrics saved to '{save_path}'.")
