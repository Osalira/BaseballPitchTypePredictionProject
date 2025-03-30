"""
Model Retraining Script

This script retrains the pitch prediction model using the original dataset
plus user feedback collected from actual outcomes. It can be scheduled to run
periodically to continuously improve the model.
"""

import os
import sys
import logging
import argparse
import time
from datetime import datetime
import pickle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_retraining.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import model training functions
from model_training import (
    load_modeling_data, prepare_train_test_data, scale_features,
    train_logistic_regression, train_decision_tree,
    train_random_forest, train_xgboost,
    evaluate_model, save_model,
    perform_feature_selection
)

# Directory paths
MODELS_DIR = os.path.join('models')

def retrain_model(model_type='xgboost'):
    """
    Retrain the model using the original dataset plus user feedback
    
    Parameters:
    -----------
    model_type : str
        Type of model to train ('xgboost', 'random_forest', 'decision_tree', 'logistic_regression')
    
    Returns:
    --------
    str
        Path to the saved model
    """
    logger.info(f"Starting retraining of {model_type} model with user feedback")
    
    # Create directories if they don't exist
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(os.path.join(MODELS_DIR, 'results'), exist_ok=True)
    
    # Load modeling data (including feedback)
    data = load_modeling_data()
    
    # Prepare train/test split
    X_train, X_test, y_train, y_test = prepare_train_test_data(data)
    
    # Perform feature selection
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
    
    # Train the selected model type
    model_path = ""
    if model_type == 'logistic_regression':
        model = train_logistic_regression(X_train_scaled, y_train)
        metrics = evaluate_model(model, X_test_scaled, y_test, "Logistic Regression")
        model_path = save_model(model, "Logistic Regression")
    
    elif model_type == 'decision_tree':
        model = train_decision_tree(X_train_selected, y_train)
        metrics = evaluate_model(model, X_test_selected, y_test, "Decision Tree")
        model_path = save_model(model, "Decision Tree")
    
    elif model_type == 'random_forest':
        model = train_random_forest(X_train_selected, y_train)
        metrics = evaluate_model(model, X_test_selected, y_test, "Random Forest")
        model_path = save_model(model, "Random Forest")
    
    elif model_type == 'xgboost':
        import xgboost as xgb
        model = train_xgboost(X_train_selected, y_train)
        # For xgboost, we need to convert the test data to DMatrix
        dtest = xgb.DMatrix(X_test_selected, label=y_test)
        metrics = evaluate_model(model, dtest, y_test, "XGBoost")
        model_path = save_model(model, "XGBoost")
    
    else:
        logger.error(f"Unsupported model type: {model_type}")
        return None
    
    # Save the metrics
    metrics_path = os.path.join(MODELS_DIR, 'results', f'{model_type}_metrics.pkl')
    with open(metrics_path, 'wb') as f:
        pickle.dump(metrics, f)
    
    logger.info(f"Model retraining complete. New model saved to {model_path}")
    return model_path

def create_schedule_file():
    """
    Create a crontab-style schedule file for automated retraining
    """
    schedule_file = os.path.join(os.path.dirname(__file__), 'retraining_schedule.txt')
    if not os.path.exists(schedule_file):
        with open(schedule_file, 'w') as f:
            f.write("# Schedule for model retraining\n")
            f.write("# Format: <minute> <hour> <day_of_month> <month> <day_of_week> <model_type>\n")
            f.write("# Use '*' for any value\n\n")
            f.write("# Example: Run XGBoost retraining every day at 2:00 AM\n")
            f.write("0 2 * * * xgboost\n\n")
            f.write("# Example: Run Random Forest retraining every Sunday at 3:00 AM\n")
            f.write("# 0 3 * * 0 random_forest\n")
        
        logger.info(f"Created schedule file at {schedule_file}")
        logger.info("Edit this file to customize your retraining schedule")

def run_scheduled_tasks():
    """
    Check the schedule file and run any tasks that are due
    """
    schedule_file = os.path.join(os.path.dirname(__file__), 'retraining_schedule.txt')
    if not os.path.exists(schedule_file):
        create_schedule_file()
        return
    
    now = datetime.now()
    current_minute = now.minute
    current_hour = now.hour
    current_day = now.day
    current_month = now.month
    current_weekday = now.weekday()  # 0-6, where 0 is Monday
    
    with open(schedule_file, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            try:
                parts = line.split()
                if len(parts) != 6:
                    continue
                
                minute, hour, day, month, weekday, model_type = parts
                
                # Check if the task is due
                if ((minute == '*' or int(minute) == current_minute) and
                    (hour == '*' or int(hour) == current_hour) and
                    (day == '*' or int(day) == current_day) and
                    (month == '*' or int(month) == current_month) and
                    (weekday == '*' or int(weekday) == current_weekday)):
                    
                    logger.info(f"Scheduled task due: Retraining {model_type} model")
                    retrain_model(model_type)
            
            except Exception as e:
                logger.error(f"Error processing scheduled task: {str(e)}")

def main():
    """Main function to execute model retraining"""
    parser = argparse.ArgumentParser(description='Retrain pitch prediction model with feedback data')
    parser.add_argument('--model', type=str, choices=['xgboost', 'random_forest', 'decision_tree', 'logistic_regression'],
                        default='xgboost', help='Type of model to train')
    parser.add_argument('--schedule', action='store_true', help='Create a schedule file for automated retraining')
    parser.add_argument('--daemon', action='store_true', help='Run as a daemon to execute scheduled tasks')
    
    args = parser.parse_args()
    
    if args.schedule:
        create_schedule_file()
    elif args.daemon:
        logger.info("Starting daemon mode to run scheduled tasks")
        while True:
            run_scheduled_tasks()
            # Sleep for a minute before checking again
            time.sleep(60)
    else:
        # Manual retraining
        retrain_model(args.model)

if __name__ == "__main__":
    main() 