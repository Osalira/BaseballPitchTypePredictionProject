#!/usr/bin/env python
"""
Firebase Bridge for Baseball Pitch Prediction

This script syncs essential data between SQLite database and Firebase,
implementing the hybrid approach for optimal performance and accessibility.
"""

import os
import json
import sqlite3
import logging
import pandas as pd
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database path
DB_PATH = os.path.join('data', 'baseball.db')

# Firebase configuration
# Note: You'll need to add your Firebase config here
# and install firebase-admin package (pip install firebase-admin)
FIREBASE_CONFIG = {
    # Add your Firebase configuration here
    # 'apiKey': 'YOUR_API_KEY',
    # 'authDomain': 'YOUR_PROJECT_ID.firebaseapp.com',
    # 'databaseURL': 'https://YOUR_PROJECT_ID.firebaseio.com',
    # 'projectId': 'YOUR_PROJECT_ID',
    # 'storageBucket': 'YOUR_PROJECT_ID.appspot.com',
    # 'messagingSenderId': 'YOUR_MESSAGING_SENDER_ID',
    # 'appId': 'YOUR_APP_ID'
}

def initialize_firebase():
    """
    Initialize Firebase connection
    
    Returns:
    --------
    firebase_admin.db
        Firebase database reference
    """
    try:
        import firebase_admin
        from firebase_admin import credentials, firestore
        
        # Check if app is already initialized
        if not firebase_admin._apps:
            # Initialize the app
            cred = credentials.Certificate("firebase-key.json")  # You'll need to create this file
            firebase_admin.initialize_app(cred)
        
        # Get Firestore database
        db = firestore.client()
        logger.info("Firebase connection initialized")
        return db
    except ImportError:
        logger.error("Firebase admin SDK not installed. Run: pip install firebase-admin")
        return None
    except Exception as e:
        logger.error(f"Error initializing Firebase: {str(e)}")
        return None

def get_model_results():
    """
    Get model results from SQLite database
    
    Returns:
    --------
    pandas.DataFrame
        Model comparison results
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        
        # Check if model results table exists
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='model_comparison'")
        if cursor.fetchone() is None:
            # Try to read from CSV file
            results_path = os.path.join('models', 'results', 'model_comparison.csv')
            if os.path.exists(results_path):
                results = pd.read_csv(results_path)
                # Save to SQLite for future access
                results.to_sql('model_comparison', conn, if_exists='replace', index=False)
            else:
                logger.error("No model comparison results found")
                return None
        else:
            # Read from SQLite
            results = pd.read_sql_query("SELECT * FROM model_comparison", conn)
        
        conn.close()
        return results
    except Exception as e:
        logger.error(f"Error getting model results: {str(e)}")
        return None

def get_features_importance():
    """
    Get feature importance from SQLite database
    
    Returns:
    --------
    pandas.DataFrame
        Feature importance data
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        
        # Check if feature importance table exists
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='feature_importance'")
        if cursor.fetchone() is None:
            # Try to read from CSV file
            fi_path = os.path.join('models', 'results', 'feature_importance.csv')
            if os.path.exists(fi_path):
                fi_data = pd.read_csv(fi_path)
                # Save to SQLite for future access
                fi_data.to_sql('feature_importance', conn, if_exists='replace', index=False)
            else:
                logger.error("No feature importance data found")
                return None
        else:
            # Read from SQLite
            fi_data = pd.read_sql_query("SELECT * FROM feature_importance", conn)
        
        conn.close()
        return fi_data
    except Exception as e:
        logger.error(f"Error getting feature importance: {str(e)}")
        return None

def get_recent_predictions(limit=100):
    """
    Get recent predictions from SQLite database
    
    Parameters:
    -----------
    limit : int
        Maximum number of predictions to return
    
    Returns:
    --------
    pandas.DataFrame
        Recent predictions
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        
        # Check if predictions table exists
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='predictions'")
        if cursor.fetchone() is None:
            # No predictions found
            logger.warning("No predictions table found")
            return None
        
        # Get recent predictions
        predictions = pd.read_sql_query(
            f"SELECT * FROM predictions ORDER BY timestamp DESC LIMIT {limit}",
            conn
        )
        
        conn.close()
        return predictions
    except Exception as e:
        logger.error(f"Error getting recent predictions: {str(e)}")
        return None

def sync_to_firebase():
    """
    Sync data from SQLite to Firebase
    """
    # Initialize Firebase
    db = initialize_firebase()
    if db is None:
        return
    
    try:
        # Connect to SQLite
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # 1. Sync model results
        model_results = get_model_results()
        if model_results is not None:
            # Convert to dictionary for Firebase
            models_data = model_results.to_dict(orient='records')
            
            # Save to Firebase
            db.collection('model_results').document('comparison').set({
                'data': models_data,
                'last_updated': datetime.now(),
                'count': len(models_data)
            })
            logger.info(f"Synced {len(models_data)} model results to Firebase")
            
            # Update sync status in SQLite
            cursor.execute(
                "INSERT OR REPLACE INTO firebase_sync (table_name, last_sync, record_count, status) VALUES (?, ?, ?, ?)",
                ('model_comparison', datetime.now().isoformat(), len(models_data), 'success')
            )
        
        # 2. Sync feature importance
        feature_importance = get_features_importance()
        if feature_importance is not None:
            # Convert to dictionary for Firebase
            fi_data = feature_importance.to_dict(orient='records')
            
            # Save to Firebase
            db.collection('model_results').document('feature_importance').set({
                'data': fi_data,
                'last_updated': datetime.now(),
                'count': len(fi_data)
            })
            logger.info(f"Synced {len(fi_data)} feature importance records to Firebase")
            
            # Update sync status in SQLite
            cursor.execute(
                "INSERT OR REPLACE INTO firebase_sync (table_name, last_sync, record_count, status) VALUES (?, ?, ?, ?)",
                ('feature_importance', datetime.now().isoformat(), len(fi_data), 'success')
            )
        
        # 3. Sync recent predictions
        predictions = get_recent_predictions(limit=100)
        if predictions is not None:
            # Convert to dictionary for Firebase
            pred_data = predictions.to_dict(orient='records')
            
            # Save to Firebase
            db.collection('predictions').document('recent').set({
                'data': pred_data,
                'last_updated': datetime.now(),
                'count': len(pred_data)
            })
            logger.info(f"Synced {len(pred_data)} recent predictions to Firebase")
            
            # Update sync status in SQLite
            cursor.execute(
                "INSERT OR REPLACE INTO firebase_sync (table_name, last_sync, record_count, status) VALUES (?, ?, ?, ?)",
                ('predictions', datetime.now().isoformat(), len(pred_data), 'success')
            )
        
        # 4. Sync metadata about available seasons
        cursor.execute("SELECT DISTINCT value FROM metadata WHERE key LIKE 'season_%'")
        seasons = [row[0] for row in cursor.fetchall()]
        
        if seasons:
            db.collection('metadata').document('seasons').set({
                'available_seasons': seasons,
                'last_updated': datetime.now()
            })
            logger.info(f"Synced metadata about {len(seasons)} available seasons")
        
        # Commit SQLite changes
        conn.commit()
        conn.close()
        
        logger.info("Firebase sync completed successfully")
        
    except Exception as e:
        logger.error(f"Error syncing to Firebase: {str(e)}")

def create_firebase_config():
    """
    Create Firebase configuration file template
    """
    config_path = 'firebase-config.json'
    
    if os.path.exists(config_path):
        logger.info(f"Firebase config file already exists at {config_path}")
        return
    
    template = {
        "apiKey": "YOUR_API_KEY",
        "authDomain": "YOUR_PROJECT_ID.firebaseapp.com",
        "databaseURL": "https://YOUR_PROJECT_ID.firebaseio.com",
        "projectId": "YOUR_PROJECT_ID",
        "storageBucket": "YOUR_PROJECT_ID.appspot.com",
        "messagingSenderId": "YOUR_MESSAGING_SENDER_ID",
        "appId": "YOUR_APP_ID"
    }
    
    with open(config_path, 'w') as f:
        json.dump(template, f, indent=2)
    
    logger.info(f"Created Firebase config template at {config_path}")
    logger.info("Please update this file with your actual Firebase configuration")

def main():
    """Main function to execute Firebase sync"""
    logger.info("Starting Firebase sync")
    
    # Create Firebase config template if needed
    create_firebase_config()
    
    # Ensure SQLite database exists
    if not os.path.exists(DB_PATH):
        logger.error(f"SQLite database not found at {DB_PATH}")
        logger.info("Please run setup_database.py first")
        return
    
    # Sync data to Firebase
    sync_to_firebase()
    
    logger.info("Firebase sync completed")
    logger.info("Next steps:")
    logger.info("1. Update firebase-config.json with your Firebase configuration")
    logger.info("2. Run this script periodically to keep Firebase in sync")
    logger.info("3. Use the data in Firebase for your Vue+Vite frontend")

if __name__ == "__main__":
    main() 