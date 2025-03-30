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
    
    # Incorporate user feedback data if available
    user_feedback_data = load_feedback_data()
    if not user_feedback_data.empty:
        data = incorporate_feedback_data(data, user_feedback_data)
    
    return data

def load_feedback_data():
    """
    Load user feedback data from the SQLite database
    
    Returns:
    --------
    pandas.DataFrame
        User feedback data
    """
    import sqlite3
    from utils import get_db_path
    
    DB_PATH = get_db_path()
    logger.info(f"Loading user feedback from SQLite database at {DB_PATH}")
    
    try:
        # Connect to SQLite
        conn = sqlite3.connect(DB_PATH)
        
        # Check if predictions table exists
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='predictions'")
        if not cursor.fetchone():
            logger.warning("Predictions table not found in SQLite")
            return pd.DataFrame()
        
        # Check if actual_pitch column exists
        cursor.execute("PRAGMA table_info(predictions)")
        columns = [col[1] for col in cursor.fetchall()]
        if 'actual_pitch' not in columns or 'was_correct' not in columns:
            logger.warning("Actual pitch data not found in predictions table")
            return pd.DataFrame()
        
        # Query predictions with actual outcomes recorded
        feedback_data = pd.read_sql_query(
            """
            SELECT * FROM predictions 
            WHERE actual_pitch IS NOT NULL
            """, 
            conn
        )
        conn.close()
        
        if feedback_data.empty:
            logger.info("No user feedback available for training")
            return pd.DataFrame()
        
        logger.info(f"Loaded {len(feedback_data)} records of user feedback")
        return feedback_data
        
    except Exception as e:
        logger.error(f"Error loading user feedback: {str(e)}")
        return pd.DataFrame()

def incorporate_feedback_data(modeling_data, feedback_data):
    """
    Incorporate user feedback into the modeling dataset
    
    Parameters:
    -----------
    modeling_data : pandas.DataFrame
        The original modeling dataset
    feedback_data : pandas.DataFrame
        User feedback data
        
    Returns:
    --------
    pandas.DataFrame
        Updated modeling dataset with user feedback
    """
    logger.info("Incorporating user feedback into modeling dataset")
    
    # Create a copy of the modeling data
    updated_data = modeling_data.copy()
    
    try:
        # Create a new dataframe with the same structure as modeling_data
        feedback_modeling = pd.DataFrame(columns=modeling_data.columns)
        
        # Map actual_pitch in feedback to is_fastball (binary target)
        fastball_types = ['FF', 'FT', 'FC', 'SI', 'FS']  # Common fastball codes
        
        for _, feedback in feedback_data.iterrows():
            # Extract game state features
            game_state = feedback.get('gameState', {})
            if isinstance(game_state, str):
                import json
                game_state = json.loads(game_state)
                
            # Extract pitcher and batter info
            pitcher_info = feedback.get('pitcherInfo', {})
            batter_info = feedback.get('batterInfo', {})
            
            if isinstance(pitcher_info, str):
                import json
                pitcher_info = json.loads(pitcher_info)
                
            if isinstance(batter_info, str):
                import json
                batter_info = json.loads(batter_info)
                
            # Create a new row
            new_row = {}
            
            # Fill in features from the feedback data
            # Game situation
            new_row['balls'] = game_state.get('balls', 0)
            new_row['strikes'] = game_state.get('strikes', 0)
            new_row['outs_when_up'] = game_state.get('outs', 0)
            new_row['inning'] = game_state.get('inning', 1)
            
            # Count type indicators
            count_type = game_state.get('countType', 'Neutral')
            new_row['hitter_count'] = 1 if count_type == 'Hitter' else 0
            new_row['pitcher_count'] = 1 if count_type == 'Pitcher' else 0
            new_row['neutral_count'] = 1 if count_type == 'Neutral' else 0
            
            # Inning stage indicators
            inning_stage = game_state.get('inningStage', 'Early')
            new_row['early_inning'] = 1 if inning_stage == 'Early' else 0
            new_row['middle_inning'] = 1 if inning_stage == 'Middle' else 0
            new_row['late_inning'] = 1 if inning_stage == 'Late' else 0
            
            # Pitcher tendencies
            new_row['pitcher_fb_pct'] = pitcher_info.get('fastballPercentage', 0.5)
            new_row['pitcher_count_fb_pct'] = pitcher_info.get('countFastballPercentage', 0.5)
            
            # Set the target variable based on the actual pitch outcome
            actual_pitch = feedback.get('actualPitch', '')
            new_row['is_fastball'] = 1 if actual_pitch in fastball_types else 0
            
            # Add other available features
            # Only include columns that exist in the modeling dataset
            for col in modeling_data.columns:
                if col not in new_row and col in feedback_data.columns:
                    new_row[col] = feedback[col]
            
            # Add the new row to the feedback modeling data
            feedback_row = pd.DataFrame([new_row])
            feedback_row = feedback_row.reindex(columns=modeling_data.columns)
            feedback_modeling = pd.concat([feedback_modeling, feedback_row], ignore_index=True)
        
        # Remove columns with all NaN values
        feedback_modeling = feedback_modeling.dropna(axis=1, how='all')
        
        # Fill remaining NaN values with sensible defaults or column means from original data
        for col in feedback_modeling.columns:
            if feedback_modeling[col].isna().all():
                if col in modeling_data.columns:
                    feedback_modeling[col] = modeling_data[col].mean()
                else:
                    feedback_modeling[col] = 0
        
        # Add feedback data to the modeling dataset
        if not feedback_modeling.empty:
            # Check if we need to add missing columns
            for col in modeling_data.columns:
                if col not in feedback_modeling.columns:
                    feedback_modeling[col] = modeling_data[col].iloc[0]
            
            # Reorder columns to match original data
            feedback_modeling = feedback_modeling[modeling_data.columns]
            
            # Combine the datasets, giving more weight to feedback data
            # Duplicate feedback data to increase its influence on the model
            weight_factor = 2  # Adjust as needed
            weighted_feedback = pd.concat([feedback_modeling] * weight_factor, ignore_index=True)
            updated_data = pd.concat([modeling_data, weighted_feedback], ignore_index=True)
            
            logger.info(f"Added {len(feedback_modeling)} feedback records with weight factor {weight_factor}")
            logger.info(f"Updated modeling data now has {len(updated_data)} records")
    
    except Exception as e:
        logger.error(f"Error incorporating feedback data: {str(e)}")
        logger.warning("Using original modeling data without feedback")
        return modeling_data
    
    return updated_data

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
        Training labels
    
    Returns:
    --------
    xgboost.Booster
        Trained XGBoost model
    """
    logger.info("Training XGBoost model")
    
    # Check if GPU is available
    try:
        import cupy  # Check for CUDA support
        gpu_available = True
        logger.info("GPU detected - enabling GPU acceleration for XGBoost")
    except ImportError:
        gpu_available = False
        logger.info("No GPU detected - using CPU for XGBoost")
    
    # Define parameters - improved for better accuracy
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'learning_rate': 0.05,  # Reduced from 0.1 for better generalization
        'max_depth': 6,        # Increased from 5 for more complex patterns
        'min_child_weight': 2,  # Increased to reduce overfitting
        'subsample': 0.9,       # Increased for better generalization
        'colsample_bytree': 0.9, # Increased for better generalization
        'seed': 42
    }
    
    # Add GPU parameters if GPU is available
    if gpu_available:
        params.update({
            'tree_method': 'gpu_hist',
            'predictor': 'gpu_predictor',
            'gpu_id': 0
        })
    
    # Create DMatrix for efficient training
    import xgboost as xgb
    dtrain = xgb.DMatrix(X_train, label=y_train)
    
    # Train with early stopping
    num_boost_round = 1000
    early_stopping_rounds = 50
    
    # Use cross-validation to find optimal number of boosting rounds
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds,
        nfold=5,
        metrics='auc',
        seed=42
    )
    
    # Get the optimal number of boosting rounds
    best_rounds = len(cv_results)
    logger.info(f"XGBoost CV suggested {best_rounds} boosting rounds")
    
    # Train final model with optimal number of boosting rounds
    final_model = xgb.train(
        params,
        dtrain,
        num_boost_round=best_rounds
    )
    
    logger.info(f"XGBoost model trained with {best_rounds} boosting rounds")
    
    return final_model

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

def perform_feature_selection(X_train, y_train, n_features=15):
    """
    Perform feature selection using mutual information
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features
    y_train : pandas.Series
        Training labels
    n_features : int
        Number of features to select
    
    Returns:
    --------
    tuple
        (selected_features, feature_importance_df) - List of selected features and DataFrame with feature importances
    """
    logger.info(f"Performing feature selection to select top {n_features} features")
    
    from sklearn.feature_selection import mutual_info_classif, SelectKBest
    
    # Calculate mutual information
    mutual_info = mutual_info_classif(X_train, y_train, random_state=42)
    
    # Create DataFrame with feature importances
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': mutual_info
    })
    
    # Sort by importance
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    # Select top features
    selected_features = feature_importance['feature'].head(n_features).tolist()
    
    logger.info(f"Selected features: {', '.join(selected_features)}")
    
    return selected_features, feature_importance

def main():
    """Main function to execute model training workflow"""
    # Create directories
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(os.path.join(MODELS_DIR, 'results'), exist_ok=True)
    
    # Load modeling data
    data = load_modeling_data()
    
    # Prepare train/test split
    X_train, X_test, y_train, y_test = prepare_train_test_data(data)
    
    # Perform feature selection (increase from 15 to 20 features to improve accuracy)
    selected_features, feature_importance_df = perform_feature_selection(X_train, y_train, n_features=20)
    
    # Update training and test data with selected features
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]
    
    # Save the feature importance for later analysis
    feature_importance_path = os.path.join(MODELS_DIR, 'results', 'feature_importance.csv')
    feature_importance_df.to_csv(feature_importance_path, index=False)
    logger.info(f"Saved feature importance to {feature_importance_path}")
    
    # Save selected features list for prediction
    features_path = os.path.join(MODELS_DIR, 'selected_features.pkl')
    with open(features_path, 'wb') as f:
        pickle.dump(selected_features, f)
    logger.info(f"Saved selected features list to {features_path}")
    
    # Scale features for appropriate models
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train_selected, X_test_selected)
    
    # Save scaler for later use in predictions
    scaler_path = os.path.join(MODELS_DIR, 'standard_scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    logger.info(f"Saved scaler to {scaler_path}")
    
    # Train and evaluate models
    model_results = []
    
    # Logistic Regression (with scaled features)
    lr_model = train_logistic_regression(X_train_scaled, y_train)
    lr_metrics = evaluate_model(lr_model, X_test_scaled, y_test, "Logistic Regression")
    save_model(lr_model, "logistic_regression")
    model_results.append(["Logistic Regression", lr_metrics['accuracy'], lr_metrics['precision'], 
                          lr_metrics['recall'], lr_metrics['f1_score'], lr_metrics['roc_auc']])
    
    # Decision Tree (without scaling)
    dt_model = train_decision_tree(X_train_selected, y_train)
    dt_metrics = evaluate_model(dt_model, X_test_selected, y_test, "Decision Tree")
    save_model(dt_model, "decision_tree")
    model_results.append(["Decision Tree", dt_metrics['accuracy'], dt_metrics['precision'], 
                          dt_metrics['recall'], dt_metrics['f1_score'], dt_metrics['roc_auc']])
    
    # Random Forest (without scaling)
    rf_model = train_random_forest(X_train_selected, y_train)
    rf_metrics = evaluate_model(rf_model, X_test_selected, y_test, "Random Forest")
    save_model(rf_model, "random_forest")
    model_results.append(["Random Forest", rf_metrics['accuracy'], rf_metrics['precision'], 
                          rf_metrics['recall'], rf_metrics['f1_score'], rf_metrics['roc_auc']])
    
    # XGBoost (without scaling)
    xgb_model = train_xgboost(X_train_selected, y_train)
    
    # For XGBoost booster, we need to convert to DMatrix for prediction
    import xgboost as xgb
    dtest = xgb.DMatrix(X_test_selected)
    y_prob_xgb = xgb_model.predict(dtest)
    y_pred_xgb = (y_prob_xgb >= 0.5).astype(int)
    
    xgb_metrics = {
        'accuracy': accuracy_score(y_test, y_pred_xgb),
        'precision': precision_score(y_test, y_pred_xgb),
        'recall': recall_score(y_test, y_pred_xgb),
        'f1': f1_score(y_test, y_pred_xgb),
        'roc_auc': roc_auc_score(y_test, y_prob_xgb)
    }
    
    # Log XGBoost metrics
    logger.info(f"XGBoost Accuracy: {xgb_metrics['accuracy']:.4f}")
    logger.info(f"XGBoost Precision: {xgb_metrics['precision']:.4f}")
    logger.info(f"XGBoost Recall: {xgb_metrics['recall']:.4f}")
    logger.info(f"XGBoost F1 Score: {xgb_metrics['f1']:.4f}")
    logger.info(f"XGBoost ROC AUC: {xgb_metrics['roc_auc']:.4f}")
    
    # Save XGBoost model
    save_model(xgb_model, "xgboost")
    model_results.append(["XGBoost", xgb_metrics['accuracy'], xgb_metrics['precision'], 
                           xgb_metrics['recall'], xgb_metrics['f1'], 
                           xgb_metrics['roc_auc']])
    
    # Create and save confusion matrices for each model
    plot_confusion_matrix(
        confusion_matrix(y_test, y_pred_xgb),
        "XGBoost",
        os.path.join(MODELS_DIR, 'results', 'xgboost_confusion_matrix.png')
    )
    
    # Save feature importance for tree-based models
    try:
        plot_feature_importance(
            xgb_model, selected_features, "XGBoost", 
            os.path.join(MODELS_DIR, 'results', 'xgboost_feature_importance.png')
        )
    except Exception as e:
        logger.warning(f"Could not generate feature importance plot for XGBoost: {str(e)}")
    
    # Compare all models and save results
    columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
    comparison_df = pd.DataFrame(model_results, columns=columns)
    comparison_path = os.path.join(MODELS_DIR, 'results', 'model_comparison.csv')
    comparison_df.to_csv(comparison_path, index=False)
    
    logger.info(f"Saved model comparison to {comparison_path}")
    logger.info("\nModel Comparison:")
    logger.info(comparison_df.to_string())
    
    return comparison_df

if __name__ == "__main__":
    main() 