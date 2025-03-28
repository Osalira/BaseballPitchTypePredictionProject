#!/usr/bin/env python
"""
Database Setup Script for Baseball Pitch Prediction

This script initializes the SQLite database structure for faster data operations.
"""

import os
import sqlite3
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database path
DB_PATH = os.path.join('data', 'baseball.db')

def create_database():
    """
    Create SQLite database with schema for baseball pitch prediction
    """
    # Ensure data directory exists
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    # Connect to database (creates if not exists)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create metadata table to track processing status
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS metadata (
        key TEXT PRIMARY KEY,
        value TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Create indices table to improve query performance
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS indices (
        table_name TEXT,
        column_name TEXT,
        index_name TEXT,
        PRIMARY KEY (table_name, column_name)
    )
    ''')
    
    # Create firebase_sync table for hybrid approach
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS firebase_sync (
        table_name TEXT PRIMARY KEY,
        last_sync DATETIME,
        record_count INTEGER,
        status TEXT
    )
    ''')
    
    # Add metadata entries
    cursor.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)", 
                 ('db_version', '1.0.0'))
    cursor.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)", 
                 ('created_at', sqlite3.datetime.datetime.now().isoformat()))
    
    # Commit changes
    conn.commit()
    conn.close()
    
    logger.info(f"Database initialized at {DB_PATH}")

def add_indices(conn, table_name, columns):
    """
    Add indices to specified columns for better performance
    
    Parameters:
    -----------
    conn : sqlite3.Connection
        Database connection
    table_name : str
        Name of the table
    columns : list
        List of column names to index
    """
    cursor = conn.cursor()
    
    for column in columns:
        index_name = f"idx_{table_name}_{column}"
        try:
            cursor.execute(f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name} ({column})")
            
            # Log the index creation
            cursor.execute(
                "INSERT OR REPLACE INTO indices (table_name, column_name, index_name) VALUES (?, ?, ?)",
                (table_name, column, index_name)
            )
            
            logger.info(f"Created index {index_name} on {table_name}.{column}")
        except sqlite3.Error as e:
            logger.error(f"Error creating index on {table_name}.{column}: {str(e)}")
    
    conn.commit()

def main():
    """Main function to setup the database"""
    logger.info("Setting up SQLite database for baseball pitch prediction")
    
    # Create database and schema
    create_database()
    
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    
    # Set up initial indices for common queries
    # These will be created when tables are imported
    tables_to_index = {
        'modeling_data_2021_to_2023': ['pitcher', 'batter', 'count', 'inning'],
        'clean_data_2021_to_2023': ['pitcher', 'batter', 'game_date', 'pitch_type']
    }
    
    # Store index configuration in metadata
    cursor = conn.cursor()
    for table, columns in tables_to_index.items():
        cursor.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            (f"index_config_{table}", ','.join(columns))
        )
    
    conn.commit()
    conn.close()
    
    logger.info("Database setup complete")
    logger.info("Next steps:")
    logger.info("1. Run 'python multi_season_demo.py' to collect and process data")
    logger.info("2. Data will be automatically stored in the SQLite database")

if __name__ == "__main__":
    main() 