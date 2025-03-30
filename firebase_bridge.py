#!/usr/bin/env python
"""
Firebase Bridge for Baseball Pitch Prediction

This script synchronizes data between the SQLite database and Firebase Firestore,
making prediction results and analysis available for the frontend application.
"""

import os
import json
import sqlite3
import logging
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import uuid  # For generating unique IDs if needed

# Load environment variables from .env file
load_dotenv()

# Firebase admin imports
try:
    import firebase_admin
    from firebase_admin import credentials
    from firebase_admin import firestore
except ImportError:
    logging.warning("Firebase admin SDK not installed. Run: pip install firebase-admin")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database path
DB_PATH = os.path.join('data', 'baseball.db')

# Firebase configuration
FIREBASE_CONFIG_PATH = 'firebase-config.json'

def initialize_firebase():
    """
    Initialize Firebase connection using service account credentials from firebase-config.json
    
    Returns:
    --------
    firestore.Client
        Firestore database client
    """
    try:
        # Check if already initialized
        if firebase_admin._apps:
            logger.info("Firebase already initialized")
            return firebase_admin.get_app()
        
        # Check if service account file exists
        if not os.path.exists(FIREBASE_CONFIG_PATH):
            logger.error(f"Firebase config file not found at {FIREBASE_CONFIG_PATH}")
            logger.info("Please ensure your firebase-config.json file is in the project root directory")
            
            # Fall back to environment variables if service account file doesn't exist
            firebase_config = {
                "projectId": os.getenv('VITE_FIREBASE_PROJECT_ID')
            }
            
            if not firebase_config["projectId"]:
                logger.error("Required Firebase environment variables not found in .env file")
                return None
            
            # Initialize without credentials (relies on environment variables)
            try:
                firebase_admin.initialize_app()
                logger.info("Firebase initialized using environment variables (limited functionality)")
            except Exception as e:
                logger.error(f"Error initializing Firebase with environment vars: {str(e)}")
                return None
        else:
            # Initialize with service account credentials
            try:
                cred = credentials.Certificate(FIREBASE_CONFIG_PATH)
                firebase_admin.initialize_app(cred)
                logger.info("Firebase initialized with service account credentials from firebase-config.json")
            except Exception as e:
                logger.error(f"Error initializing Firebase with service account: {str(e)}")
                return None
        
        # Get Firestore client
        db = firestore.client()
        return db
    
    except Exception as e:
        logger.error(f"Error initializing Firebase: {str(e)}")
        return None

def get_model_results():
    """
    Get model comparison results from SQLite
    
    Returns:
    --------
    pandas.DataFrame
        Model comparison results
    """
    try:
        # Connect to SQLite
        conn = sqlite3.connect(DB_PATH)
        
        # Check if table exists
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='model_comparison'")
        if not cursor.fetchone():
            logger.info("Model comparison table not found in SQLite, trying CSV file")
            csv_path = os.path.join('models', 'results', 'model_comparison.csv')
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                logger.info(f"Loaded model comparison from {csv_path}")
                
                # Ensure there's an ID column
                if 'id' not in df.columns:
                    logger.info("Adding ID column to model comparison data")
                    df['id'] = [str(uuid.uuid4()) for _ in range(len(df))]
                
                return df
            else:
                logger.warning("No model comparison data found")
                return pd.DataFrame()
        
        # Query from SQLite
        df = pd.read_sql_query("SELECT * FROM model_comparison", conn)
        conn.close()
        
        # Ensure there's an ID column
        if 'id' not in df.columns:
            logger.info("Adding ID column to model comparison data")
            df['id'] = [str(uuid.uuid4()) for _ in range(len(df))]
        
        return df
    
    except Exception as e:
        logger.error(f"Error getting model results: {str(e)}")
        return pd.DataFrame()

def get_features_importance():
    """
    Get feature importance data from SQLite
    
    Returns:
    --------
    pandas.DataFrame
        Feature importance data
    """
    try:
        # Connect to SQLite
        conn = sqlite3.connect(DB_PATH)
        
        # Check if table exists
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='feature_importance'")
        if not cursor.fetchone():
            logger.info("Feature importance table not found in SQLite, trying CSV file")
            csv_path = os.path.join('models', 'results', 'feature_importance.csv')
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                logger.info(f"Loaded feature importance from {csv_path}")
                return df
            else:
                logger.warning("No feature importance data found")
                return pd.DataFrame()
        
        # Query from SQLite
        df = pd.read_sql_query("SELECT * FROM feature_importance", conn)
        conn.close()
        
        return df
    
    except Exception as e:
        logger.error(f"Error getting feature importance: {str(e)}")
        return pd.DataFrame()

def get_recent_predictions(limit=100):
    """
    Get recent predictions from SQLite
    
    Parameters:
    -----------
    limit : int
        Maximum number of predictions to retrieve
    
    Returns:
    --------
    pandas.DataFrame
        Recent predictions data
    """
    try:
        # Connect to SQLite
        conn = sqlite3.connect(DB_PATH)
        
        # Check if table exists
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='predictions'")
        if not cursor.fetchone():
            logger.warning("Predictions table not found in SQLite")
            return pd.DataFrame()
        
        # Query from SQLite
        df = pd.read_sql_query(
            f"SELECT * FROM predictions ORDER BY timestamp DESC LIMIT {limit}", 
            conn
        )
        conn.close()
        
        # Ensure there's an ID column
        if not df.empty and 'id' not in df.columns:
            logger.info("Adding ID column to predictions data")
            df['id'] = [str(uuid.uuid4()) for _ in range(len(df))]
        
        return df
    
    except Exception as e:
        logger.error(f"Error getting recent predictions: {str(e)}")
        return pd.DataFrame()

def sync_to_firebase():
    """
    Synchronize data from SQLite to Firebase Firestore.
    This ensures that the frontend and backend data stays consistent.
    """
    db = sqlite3.connect(DB_PATH)
    firebase_app = initialize_firebase()
    db_firestore = firestore.client(firebase_app)
    
    # Sync model results to Firestore
    try:
        model_results = get_model_results()
        if not model_results.empty:
            # Check if 'id' column exists, add it if it doesn't
            if 'id' not in model_results.columns:
                logger.warning("No 'id' column found in model results. Adding generated IDs.")
                model_results['id'] = [str(uuid.uuid4()) for _ in range(len(model_results))]
            
            results_collection = db_firestore.collection('model_results')
            
            # Check which results already exist in Firestore
            existing_results = []
            docs = results_collection.stream()
            for doc in docs:
                result_data = doc.to_dict()
                if 'id' in result_data:
                    existing_results.append(result_data['id'])
            
            # Add any new results
            for index, result in model_results.iterrows():
                try:
                    # First convert the Series to a dictionary
                    result_dict = result.to_dict()
                    
                    # Check if ID exists in the dictionary
                    if 'id' not in result_dict:
                        result_dict['id'] = str(uuid.uuid4())
                    
                    result_id = result_dict['id']
                    if result_id not in existing_results:
                        # Add to Firestore
                        results_collection.add(result_dict)
                        logger.info(f"Added model result {result_id} to Firestore")
                except Exception as e:
                    logger.error(f"Error processing model result: {e}")
                    continue  # Skip this record and continue with others
    except Exception as e:
        logger.error(f"Error syncing model results: {e}")
    
    # Sync predictions to Firestore
    try:
        predictions = get_recent_predictions()
        if not predictions.empty:
            # Check if 'id' column exists, add it if it doesn't
            if 'id' not in predictions.columns:
                logger.warning("No 'id' column found in predictions. Adding generated IDs.")
                predictions['id'] = [str(uuid.uuid4()) for _ in range(len(predictions))]
                
            predictions_collection = db_firestore.collection('predictions')
            
            # Check which predictions already exist in Firestore
            existing_predictions = []
            docs = predictions_collection.stream()
            for doc in docs:
                prediction_data = doc.to_dict()
                if 'id' in prediction_data:
                    existing_predictions.append(prediction_data['id'])
            
            # Add any new predictions
            for index, prediction in predictions.iterrows():
                try:
                    # First convert the Series to a dictionary
                    prediction_dict = prediction.to_dict()
                    
                    # Check if ID exists in the dictionary
                    if 'id' not in prediction_dict:
                        prediction_dict['id'] = str(uuid.uuid4())
                    
                    prediction_id = prediction_dict['id']
                    if prediction_id not in existing_predictions:
                        # Add to Firestore
                        predictions_collection.add(prediction_dict)
                        logger.info(f"Added prediction {prediction_id} to Firestore")
                except Exception as e:
                    logger.error(f"Error processing prediction: {e}")
                    continue  # Skip this record and continue with others
    except Exception as e:
        logger.error(f"Error syncing predictions: {e}")
    
    # Sync verified outcomes from Firestore back to SQLite
    sync_outcomes_from_firebase()

def sync_outcomes_from_firebase():
    """
    Synchronize actual outcome data from Firestore to SQLite.
    This allows the system to learn from user feedback.
    """
    db = sqlite3.connect(DB_PATH)
    cursor = db.cursor()
    
    # Check if predictions table exists
    cursor.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name='predictions'
    """)
    if not cursor.fetchone():
        logger.info("Predictions table does not exist in SQLite, skipping outcome sync")
        return
    
    # Check if the actualPitch column exists, add it if it doesn't
    cursor.execute("PRAGMA table_info(predictions)")
    columns = [col[1] for col in cursor.fetchall()]
    
    if 'actual_pitch' not in columns:
        cursor.execute("ALTER TABLE predictions ADD COLUMN actual_pitch TEXT")
        logger.info("Added actual_pitch column to predictions table")
    
    if 'was_correct' not in columns:
        cursor.execute("ALTER TABLE predictions ADD COLUMN was_correct BOOLEAN")
        logger.info("Added was_correct column to predictions table")
    
    # Get predictions from Firestore with actual outcomes
    firebase_app = initialize_firebase()
    db_firestore = firestore.client(firebase_app)
    predictions_collection = db_firestore.collection('predictions')
    
    # Query for predictions that have actual outcomes recorded
    verified_predictions = predictions_collection.where('actualPitch', '!=', None).stream()
    
    update_count = 0
    for doc in verified_predictions:
        prediction = doc.to_dict()
        if 'id' in prediction and 'actualPitch' in prediction:
            prediction_id = prediction['id']
            actual_pitch = prediction['actualPitch']
            was_correct = prediction.get('wasCorrect', False)
            
            # Update the prediction in SQLite with the actual outcome
            cursor.execute("""
                UPDATE predictions 
                SET actual_pitch = ?, was_correct = ?
                WHERE id = ? AND (actual_pitch IS NULL OR actual_pitch != ?)
            """, (actual_pitch, was_correct, prediction_id, actual_pitch))
            
            if cursor.rowcount > 0:
                update_count += 1
                logger.info(f"Updated prediction {prediction_id} with actual pitch {actual_pitch}")
    
    db.commit()
    logger.info(f"Synced {update_count} verified outcomes from Firestore to SQLite")
    db.close()

def create_firebase_config():
    """
    Create Firebase configuration file template using environment variables
    """
    config_path = FIREBASE_CONFIG_PATH
    
    if os.path.exists(config_path):
        logger.info(f"Firebase config file already exists at {config_path}")
        return
    
    # Get values from environment variables
    template = {
        "type": "service_account",
        "project_id": os.getenv('VITE_FIREBASE_PROJECT_ID', 'YOUR_PROJECT_ID'),
        "private_key_id": "YOUR_PRIVATE_KEY_ID",
        "private_key": "YOUR_PRIVATE_KEY",
        "client_email": "YOUR_CLIENT_EMAIL",
        "client_id": "YOUR_CLIENT_ID",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_x509_cert_url": "YOUR_CLIENT_CERT_URL",
        # Include other Firebase configs for reference
        "apiKey": os.getenv('VITE_FIREBASE_API_KEY', 'YOUR_API_KEY'),
        "authDomain": os.getenv('VITE_FIREBASE_AUTH_DOMAIN', 'YOUR_AUTH_DOMAIN'),
        "storageBucket": os.getenv('VITE_FIREBASE_STORAGE_BUCKET', 'YOUR_STORAGE_BUCKET'),
        "messagingSenderId": os.getenv('VITE_FIREBASE_MESSAGING_SENDER_ID', 'YOUR_MESSAGING_SENDER_ID'),
        "appId": os.getenv('VITE_FIREBASE_APP_ID', 'YOUR_APP_ID')
    }
    
    with open(config_path, 'w') as f:
        json.dump(template, f, indent=2)
    
    logger.info(f"Created Firebase config template at {config_path}")
    logger.info("Please update this file with your actual Firestore service account credentials")
    logger.info("Or set the GOOGLE_APPLICATION_CREDENTIALS environment variable to your service account JSON file path")

def main():
    """Main function to execute Firebase sync"""
    logger.info("Starting Firebase sync")
    
    # Ensure the Firebase config exists
    if not os.path.exists(FIREBASE_CONFIG_PATH):
        logger.warning(f"Firebase config file not found at {FIREBASE_CONFIG_PATH}")
        logger.info("Will attempt to use environment variables, but some functionality may be limited")
    
    # Ensure SQLite database exists
    if not os.path.exists(DB_PATH):
        logger.error(f"SQLite database not found at {DB_PATH}")
        logger.info("Please run setup_database.py first")
        return
    
    # Initialize Firebase and sync data
    db = initialize_firebase()
    if db:
        try:
            sync_to_firebase()
            logger.info("Firebase sync process completed successfully")
        except Exception as e:
            logger.error(f"Error during Firebase sync: {str(e)}")
    else:
        logger.error("Failed to initialize Firebase. Cannot sync data.")
    
    logger.info("Next steps:")
    logger.info("1. The pitch prediction frontend can now access this data")
    logger.info("2. Run this script periodically to keep Firebase in sync with your SQLite database")

if __name__ == "__main__":
    main() 