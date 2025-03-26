"""
Real-time Pitch Prediction Script

This script loads a trained model and provides functions for
making real-time predictions of pitch types during a game.
"""

import os
import pandas as pd
import numpy as np
import pickle
import glob
import logging
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pitch_prediction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Directory paths
MODELS_DIR = os.path.join('models')

def load_latest_model(model_type='xgboost'):
    """
    Load the most recent trained model of the specified type
    
    Parameters:
    -----------
    model_type : str
        Type of model to load ('xgboost', 'random_forest', 'decision_tree', 'logistic_regression')
    
    Returns:
    --------
    tuple
        (model, scaler) - The loaded model and scaler
    """
    # Normalize model_type for filename matching
    model_type = model_type.lower().replace(' ', '_')
    
    # Find the most recent model file of the specified type
    model_files = glob.glob(os.path.join(MODELS_DIR, f"{model_type}_*.pkl"))
    
    if not model_files:
        logger.error(f"No {model_type} model found. Run model_training.py first.")
        raise FileNotFoundError(f"No {model_type} model available")
    
    # Sort by modification time to get the most recent
    most_recent_model = max(model_files, key=os.path.getmtime)
    logger.info(f"Loading model from: {most_recent_model}")
    
    # Load the model
    with open(most_recent_model, 'rb') as f:
        model = pickle.load(f)
    
    # Load the scaler if this is a model that needs scaled inputs
    scaler = None
    if model_type == 'logistic_regression':
        scaler_path = os.path.join(MODELS_DIR, 'standard_scaler.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            logger.info(f"Loaded scaler from: {scaler_path}")
        else:
            logger.warning("No scaler found, but model requires scaling. Results may be inaccurate.")
    
    return model, scaler

def prepare_input_features(feature_values):
    """
    Prepare input features for prediction
    
    Parameters:
    -----------
    feature_values : dict
        Dictionary with feature names and values
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with features prepared for the model
    """
    # Convert to DataFrame with a single row
    df = pd.DataFrame([feature_values])
    
    # Handle missing values if any
    for col in df.columns:
        if pd.isna(df[col]).any():
            logger.warning(f"Missing value for feature: {col}. Using 0 as default.")
            df[col] = df[col].fillna(0)
    
    return df

def predict_pitch_type(model, features, scaler=None):
    """
    Predict the type of the next pitch
    
    Parameters:
    -----------
    model : object
        Trained model
    features : pandas.DataFrame
        Input features
    scaler : object, optional
        Scaler for feature normalization
    
    Returns:
    --------
    tuple
        (prediction, probability) - Predicted class and confidence
    """
    # Scale features if scaler is provided
    if scaler is not None:
        features_scaled = scaler.transform(features)
        features_to_use = pd.DataFrame(features_scaled, columns=features.columns)
    else:
        features_to_use = features
    
    # Make prediction
    prediction = model.predict(features_to_use)[0]
    
    # Get probability if available
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(features_to_use)[0]
        probability = probabilities[1] if prediction == 1 else probabilities[0]
    else:
        probability = None
    
    return prediction, probability

def get_pitch_category(prediction):
    """
    Get the pitch category label from the prediction
    
    Parameters:
    -----------
    prediction : int
        Binary prediction (0 or 1)
    
    Returns:
    --------
    str
        Pitch category label ('Fastball' or 'Offspeed')
    """
    return 'Fastball' if prediction == 1 else 'Offspeed'

def interactive_prediction():
    """
    Interactive console application for pitch prediction
    """
    print("\n=== Baseball Pitch Prediction System ===\n")
    
    # Load model
    model_choices = ['xgboost', 'random_forest', 'decision_tree', 'logistic_regression']
    print("Available model types:")
    for i, model_type in enumerate(model_choices, 1):
        print(f"{i}. {model_type.replace('_', ' ').title()}")
    
    choice = input("\nSelect model type (1-4) [default=1]: ").strip()
    model_type = model_choices[int(choice) - 1] if choice and choice.isdigit() and 1 <= int(choice) <= 4 else model_choices[0]
    
    try:
        model, scaler = load_latest_model(model_type)
        print(f"\nLoaded {model_type.replace('_', ' ').title()} model successfully.")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return
    
    while True:
        print("\n--- New Prediction ---\n")
        
        # Gather input features
        features = {}
        
        # Game situation
        print("\nGame Situation:")
        features['balls'] = int(input("Balls (0-3): ").strip() or "0")
        features['strikes'] = int(input("Strikes (0-2): ").strip() or "0")
        features['outs_when_up'] = int(input("Outs (0-2): ").strip() or "0")
        features['inning'] = int(input("Inning (1-9+): ").strip() or "1")
        
        # Count types
        count = f"{features['balls']}-{features['strikes']}"
        hitter_counts = ['1-0', '2-0', '3-0', '2-1', '3-1', '3-2']
        pitcher_counts = ['0-1', '0-2', '1-2', '2-2']
        
        features['hitter_count'] = 1 if count in hitter_counts else 0
        features['pitcher_count'] = 1 if count in pitcher_counts else 0
        features['neutral_count'] = 1 if count not in hitter_counts + pitcher_counts else 0
        
        # Inning stage
        features['early_inning'] = 1 if features['inning'] <= 3 else 0
        features['middle_inning'] = 1 if 3 < features['inning'] <= 6 else 0
        features['late_inning'] = 1 if features['inning'] > 6 else 0
        
        # Pitcher tendencies
        print("\nPitcher Information:")
        features['pitcher_fb_pct'] = float(input("Pitcher's fastball percentage (0-1): ").strip() or "0.6")
        features['pitcher_count_fb_pct'] = float(input(f"Pitcher's fastball percentage in {count} count (0-1): ").strip() or "0.6")
        features['pitcher_hitter_count_fb_pct'] = float(input("Pitcher's fastball percentage in current count type (0-1): ").strip() or "0.6")
        
        # Catcher tendencies (if available)
        include_catcher = input("\nInclude catcher information? (y/n) [default=n]: ").strip().lower() == 'y'
        if include_catcher:
            features['catcher_fb_pct'] = float(input("Catcher's fastball percentage (0-1): ").strip() or "0.6")
            features['catcher_count_fb_pct'] = float(input(f"Catcher's fastball percentage in {count} count (0-1): ").strip() or "0.6")
        
        # Batter tendencies
        print("\nBatter Information:")
        features['batter_success_vs_fb'] = float(input("Batter's success rate vs fastballs (0-1): ").strip() or "0.3")
        features['batter_success_vs_os'] = float(input("Batter's success rate vs offspeed (0-1): ").strip() or "0.25")
        features['batter_fb_os_diff'] = features['batter_success_vs_fb'] - features['batter_success_vs_os']
        
        # Sequence information
        print("\nPitch Sequence:")
        first_pitch = input("Is this the first pitch of the at-bat? (y/n) [default=n]: ").strip().lower() == 'y'
        features['first_pitch'] = 1 if first_pitch else 0
        
        if not first_pitch:
            prev_fastball = input("Was the previous pitch a fastball? (y/n) [default=y]: ").strip().lower() != 'n'
            features['prev_is_fastball'] = 1 if prev_fastball else 0
            
            consecutive = int(input("How many consecutive similar pitches so far? (0+) [default=0]: ").strip() or "0")
            features['consecutive_counter'] = consecutive
        else:
            features['prev_is_fastball'] = 0
            features['consecutive_counter'] = 0
        
        # Prepare features for prediction
        input_features = prepare_input_features(features)
        
        # Make prediction
        prediction, probability = predict_pitch_type(model, input_features, scaler)
        pitch_category = get_pitch_category(prediction)
        
        # Display result
        print("\n=== Prediction Result ===")
        print(f"Next pitch prediction: {pitch_category}")
        if probability is not None:
            print(f"Confidence: {probability:.2%}")
        
        # Ask if user wants to make another prediction
        if input("\nMake another prediction? (y/n) [default=y]: ").strip().lower() == 'n':
            break
    
    print("\nThank you for using the Baseball Pitch Prediction System!")

def predict_from_command_line(args):
    """
    Make a prediction based on command line arguments
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command line arguments
    """
    # Load model
    model, scaler = load_latest_model(args.model_type)
    
    # Prepare features
    features = {
        'balls': args.balls,
        'strikes': args.strikes,
        'outs_when_up': args.outs,
        'inning': args.inning,
        'pitcher_fb_pct': args.pitcher_fb_pct,
        'batter_success_vs_fb': args.batter_vs_fb,
        'batter_success_vs_os': args.batter_vs_os
    }
    
    # Add count features
    count = f"{args.balls}-{args.strikes}"
    hitter_counts = ['1-0', '2-0', '3-0', '2-1', '3-1', '3-2']
    pitcher_counts = ['0-1', '0-2', '1-2', '2-2']
    
    features['hitter_count'] = 1 if count in hitter_counts else 0
    features['pitcher_count'] = 1 if count in pitcher_counts else 0
    features['neutral_count'] = 1 if count not in hitter_counts + pitcher_counts else 0
    
    # Add inning features
    features['early_inning'] = 1 if args.inning <= 3 else 0
    features['middle_inning'] = 1 if 3 < args.inning <= 6 else 0
    features['late_inning'] = 1 if args.inning > 6 else 0
    
    # Add pitch sequence features
    features['first_pitch'] = 1 if args.first_pitch else 0
    features['prev_is_fastball'] = 1 if args.prev_fastball else 0
    features['consecutive_counter'] = args.consecutive
    
    # Add remaining features
    features['pitcher_count_fb_pct'] = args.pitcher_count_fb_pct
    features['pitcher_hitter_count_fb_pct'] = args.pitcher_hitter_fb_pct
    features['batter_fb_os_diff'] = args.batter_vs_fb - args.batter_vs_os
    
    # Prepare features for prediction
    input_features = prepare_input_features(features)
    
    # Make prediction
    prediction, probability = predict_pitch_type(model, input_features, scaler)
    pitch_category = get_pitch_category(prediction)
    
    # Display result
    print(f"\nPrediction: {pitch_category}")
    if probability is not None:
        print(f"Confidence: {probability:.2%}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Predict baseball pitch type (fastball or offspeed)')
    
    # Add arguments
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    parser.add_argument('--model-type', type=str, default='xgboost', 
                      choices=['xgboost', 'random_forest', 'decision_tree', 'logistic_regression'],
                      help='Type of model to use for prediction')
    
    # Game situation
    parser.add_argument('--balls', type=int, default=0, choices=range(4), help='Number of balls (0-3)')
    parser.add_argument('--strikes', type=int, default=0, choices=range(3), help='Number of strikes (0-2)')
    parser.add_argument('--outs', type=int, default=0, choices=range(3), help='Number of outs (0-2)')
    parser.add_argument('--inning', type=int, default=1, help='Inning number')
    
    # Pitcher info
    parser.add_argument('--pitcher-fb-pct', type=float, default=0.6, help='Pitcher fastball percentage (0-1)')
    parser.add_argument('--pitcher-count-fb-pct', type=float, default=0.6, 
                      help='Pitcher fastball percentage in current count (0-1)')
    parser.add_argument('--pitcher-hitter-fb-pct', type=float, default=0.6,
                      help='Pitcher fastball percentage in current count type (0-1)')
    
    # Batter info
    parser.add_argument('--batter-vs-fb', type=float, default=0.3, 
                      help='Batter success rate vs fastballs (0-1)')
    parser.add_argument('--batter-vs-os', type=float, default=0.25,
                      help='Batter success rate vs offspeed (0-1)')
    
    # Sequence info
    parser.add_argument('--first-pitch', action='store_true', help='Is this the first pitch of the at-bat')
    parser.add_argument('--prev-fastball', action='store_true', help='Was the previous pitch a fastball')
    parser.add_argument('--consecutive', type=int, default=0, 
                      help='Number of consecutive similar pitches so far')
    
    args = parser.parse_args()
    
    try:
        if args.interactive:
            interactive_prediction()
        else:
            predict_from_command_line(args)
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 