"""
Model Training Script for Baseball Pitch Prediction

This script trains various machine learning models to predict
whether the next pitch will be a fastball or offspeed.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import logging
import glob
from datetime import datetime

# Machine Learning imports
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve
)
import xgboost as xgb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Directory paths
PROCESSED_DATA_DIR = os.path.join('data', 'processed')
MODELS_DIR = os.path.join('models')
RESULTS_DIR = os.path.join('models', 'results')

def load_modeling_data():
    """
    Load the most recent modeling-ready dataset
    
    Returns:
    --------
    pandas.DataFrame
        The prepared modeling dataset
    """
    # Find the most recent modeling data file
    modeling_files = glob.glob(os.path.join(PROCESSED_DATA_DIR, "modeling_data_*.csv"))
    
    if not modeling_files:
        logger.error("No modeling data found. Run data_preprocessing.py first.")
        raise FileNotFoundError("No modeling data available")
    
    # Sort by modification time to get the most recent
    most_recent_file = max(modeling_files, key=os.path.getmtime)
    logger.info(f"Loading modeling data from: {most_recent_file}")
    
    # Load the data
    data = pd.read_csv(most_recent_file)
    logger.info(f"Loaded {len(data)} samples with {len(data.columns)} columns")
    
    return data

def prepare_train_test_data(data, test_size=0.2, random_state=42):
    """
    Split data into features and target, and into training and test sets
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The modeling dataset
    test_size : float
        Proportion of data to use for testing
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test) - split data
    """
    logger.info("Preparing training and test data")
    
    # Separate features and target
    if 'is_fastball' not in data.columns:
        logger.error("Target variable 'is_fastball' not found in dataset")
        raise ValueError("Target variable missing")
    
    y = data['is_fastball']
    X = data.drop('is_fastball', axis=1)
    
    logger.info(f"Features: {X.columns.tolist()}")
    
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logger.info(f"Training set: {X_train.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test):
    """
    Scale numerical features
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features
    X_test : pandas.DataFrame
        Test features
    
    Returns:
    --------
    tuple
        (X_train_scaled, X_test_scaled, scaler) - scaled data and scaler object
    """
    logger.info("Scaling features")
    
    # Initialize the scaler
    scaler = StandardScaler()
    
    # Fit on training data and transform both training and test data
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrames with column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    return X_train_scaled, X_test_scaled, scaler

def train_logistic_regression(X_train, y_train):
    """
    Train a Logistic Regression model
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features
    y_train : pandas.Series
        Training target
    
    Returns:
    --------
    sklearn.linear_model.LogisticRegression
        Trained Logistic Regression model
    """
    logger.info("Training Logistic Regression model")
    
    # Define parameter grid for grid search
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'class_weight': [None, 'balanced']
    }
    
    # Initialize model
    logreg = LogisticRegression(random_state=42, max_iter=1000)
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        logreg, param_grid, cv=5, scoring='accuracy', n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_logreg = grid_search.best_estimator_
    logger.info(f"Logistic Regression best parameters: {grid_search.best_params_}")
    
    return best_logreg

def train_decision_tree(X_train, y_train):
    """
    Train a Decision Tree model
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features
    y_train : pandas.Series
        Training target
    
    Returns:
    --------
    sklearn.tree.DecisionTreeClassifier
        Trained Decision Tree model
    """
    logger.info("Training Decision Tree model")
    
    # Define parameter grid for grid search
    param_grid = {
        'max_depth': [3, 5, 7, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': [None, 'balanced']
    }
    
    # Initialize model
    dt = DecisionTreeClassifier(random_state=42)
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        dt, param_grid, cv=5, scoring='accuracy', n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_dt = grid_search.best_estimator_
    logger.info(f"Decision Tree best parameters: {grid_search.best_params_}")
    
    return best_dt

def train_random_forest(X_train, y_train):
    """
    Train a Random Forest model
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features
    y_train : pandas.Series
        Training target
    
    Returns:
    --------
    sklearn.ensemble.RandomForestClassifier
        Trained Random Forest model
    """
    logger.info("Training Random Forest model")
    
    # Define parameter grid for grid search
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 20, None],
        'min_samples_split': [2, 5, 10],
        'class_weight': [None, 'balanced']
    }
    
    # Initialize model
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_rf = grid_search.best_estimator_
    logger.info(f"Random Forest best parameters: {grid_search.best_params_}")
    
    return best_rf

def train_xgboost(X_train, y_train):
    """
    Train an XGBoost model
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features
    y_train : pandas.Series
        Training target
    
    Returns:
    --------
    xgboost.XGBClassifier
        Trained XGBoost model
    """
    logger.info("Training XGBoost model")
    
    # Define parameter grid for grid search
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    # Initialize model
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        xgb_model, param_grid, cv=3, scoring='accuracy', n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_xgb = grid_search.best_estimator_
    logger.info(f"XGBoost best parameters: {grid_search.best_params_}")
    
    return best_xgb

def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate model performance on test data
    
    Parameters:
    -----------
    model : object
        Trained model
    X_test : pandas.DataFrame
        Test features
    y_test : pandas.Series
        Test target
    model_name : str
        Name of the model for logging
    
    Returns:
    --------
    dict
        Dictionary with evaluation metrics
    """
    logger.info(f"Evaluating {model_name} model")
    
    # Predict on test data
    y_pred = model.predict(X_test)
    
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_pred_proba = None
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # Precision, recall, f1-score
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # ROC-AUC (if probability predictions available)
    if y_pred_proba is not None:
        roc_auc = roc_auc_score(y_test, y_pred_proba)
    else:
        roc_auc = None
    
    # Log results
    logger.info(f"{model_name} Accuracy: {accuracy:.4f}")
    logger.info(f"{model_name} Precision: {precision:.4f}")
    logger.info(f"{model_name} Recall: {recall:.4f}")
    logger.info(f"{model_name} F1 Score: {f1:.4f}")
    if roc_auc:
        logger.info(f"{model_name} ROC AUC: {roc_auc:.4f}")
    
    # Store metrics in a dict
    metrics = {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm
    }
    
    return metrics

def plot_confusion_matrix(cm, model_name, save_path=None):
    """
    Plot confusion matrix
    
    Parameters:
    -----------
    cm : numpy.ndarray
        Confusion matrix
    model_name : str
        Name of the model
    save_path : str, optional
        Path to save the plot
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=['Offspeed', 'Fastball'],
        yticklabels=['Offspeed', 'Fastball']
    )
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name}')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix plot saved to {save_path}")
    
    plt.close()

def plot_feature_importance(model, feature_names, model_name, save_path=None):
    """
    Plot feature importance for tree-based models
    
    Parameters:
    -----------
    model : object
        Trained model
    feature_names : list
        List of feature names
    model_name : str
        Name of the model
    save_path : str, optional
        Path to save the plot
    """
    # Get feature importances
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif model_name == 'Logistic Regression' and hasattr(model, 'coef_'):
        importances = abs(model.coef_[0])  # Take absolute values for logistic regression
    else:
        logger.warning(f"No feature importance available for {model_name}")
        return
    
    # Sort features by importance
    indices = np.argsort(importances)[::-1]
    
    # Plot
    plt.figure(figsize=(12, 8))
    plt.title(f'Feature Importance - {model_name}')
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Feature importance plot saved to {save_path}")
    
    plt.close()

def save_model(model, model_name):
    """
    Save trained model to disk
    
    Parameters:
    -----------
    model : object
        Trained model
    model_name : str
        Name of the model
    
    Returns:
    --------
    str
        Path where model was saved
    """
    # Create timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create filename
    filename = f"{model_name.lower().replace(' ', '_')}_{timestamp}.pkl"
    filepath = os.path.join(MODELS_DIR, filename)
    
    # Save the model
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    logger.info(f"Saved {model_name} model to {filepath}")
    
    return filepath

def main():
    """Main function to execute model training workflow"""
    # Create directories if they don't exist
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Load modeling data
    data = load_modeling_data()
    
    # Prepare training and test data
    X_train, X_test, y_train, y_test = prepare_train_test_data(data)
    
    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Save feature names for later use
    feature_names = X_train.columns.tolist()
    
    # Train models
    models = {}
    
    # 1. Logistic Regression (baseline model)
    logreg = train_logistic_regression(X_train_scaled, y_train)
    models['Logistic Regression'] = logreg
    
    # 2. Decision Tree (interpretable model)
    dt = train_decision_tree(X_train, y_train)  # No scaling needed for tree models
    models['Decision Tree'] = dt
    
    # 3. Random Forest
    rf = train_random_forest(X_train, y_train)  # No scaling needed for tree models
    models['Random Forest'] = rf
    
    # 4. XGBoost
    xgb_model = train_xgboost(X_train, y_train)  # No scaling needed for tree models
    models['XGBoost'] = xgb_model
    
    # Evaluate models and save results
    results = []
    
    for model_name, model in models.items():
        logger.info(f"Processing {model_name} model")
        
        # Use scaled features for Logistic Regression
        if model_name == 'Logistic Regression':
            metrics = evaluate_model(model, X_test_scaled, y_test, model_name)
        else:
            metrics = evaluate_model(model, X_test, y_test, model_name)
        
        results.append(metrics)
        
        # Plot confusion matrix
        cm_path = os.path.join(RESULTS_DIR, f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png")
        plot_confusion_matrix(metrics['confusion_matrix'], model_name, cm_path)
        
        # Plot feature importance
        fi_path = os.path.join(RESULTS_DIR, f"{model_name.lower().replace(' ', '_')}_feature_importance.png")
        plot_feature_importance(model, feature_names, model_name, fi_path)
        
        # Save the model
        saved_path = save_model(model, model_name)
        
        # For decision tree, also save a visualization
        if model_name == 'Decision Tree':
            dt_plot_path = os.path.join(RESULTS_DIR, 'decision_tree_visualization.png')
            plt.figure(figsize=(20, 10))
            plot_tree(model, feature_names=feature_names, filled=True, rounded=True, proportion=True)
            plt.savefig(dt_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Decision Tree visualization saved to {dt_plot_path}")
    
    # Save scaler for future use
    scaler_path = os.path.join(MODELS_DIR, 'standard_scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    logger.info(f"Saved scaler to {scaler_path}")
    
    # Compile results into a comparison table
    results_df = pd.DataFrame([
        {
            'Model': r['model_name'],
            'Accuracy': r['accuracy'],
            'Precision': r['precision'],
            'Recall': r['recall'],
            'F1 Score': r['f1_score'],
            'ROC AUC': r['roc_auc']
        }
        for r in results
    ])
    
    # Save results to CSV
    results_path = os.path.join(RESULTS_DIR, 'model_comparison.csv')
    results_df.to_csv(results_path, index=False)
    logger.info(f"Saved model comparison to {results_path}")
    
    # Print final comparison
    logger.info("\nModel Comparison:")
    logger.info(results_df.to_string(index=False))

if __name__ == "__main__":
    main() 