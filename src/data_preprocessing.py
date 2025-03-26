"""
Data Preprocessing Script for Baseball Pitch Prediction

This script processes the raw MLB Statcast data to create features
needed for the pitch prediction model.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import glob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_preprocessing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Directory paths
RAW_DATA_DIR = os.path.join('data', 'raw')
PROCESSED_DATA_DIR = os.path.join('data', 'processed')

def load_data(file_path):
    """
    Load the classified pitch data
    
    Parameters:
    -----------
    file_path : str
        Path to the processed data CSV file
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the pitch data
    """
    logger.info(f"Loading data from {file_path}")
    try:
        data = pd.read_csv(file_path)
        logger.info(f"Successfully loaded {len(data)} records")
        return data
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def clean_data(data):
    """
    Clean the dataset by handling missing values and filtering
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Raw pitch data
    
    Returns:
    --------
    pandas.DataFrame
        Cleaned pitch data
    """
    logger.info(f"Initial data shape: {data.shape}")
    
    # Remove rows with missing essential values
    essential_columns = ['pitch_type', 'game_date', 'pitcher', 'batter', 'balls', 'strikes']
    data = data.dropna(subset=essential_columns)
    logger.info(f"After dropping rows with missing essentials: {data.shape}")
    
    # Filter out rows with pitch_type as null or 'UN' (unknown)
    data = data[data['pitch_type'].notna() & (data['pitch_type'] != 'UN')]
    logger.info(f"After filtering out unknown pitch types: {data.shape}")
    
    # Make sure numeric columns are properly typed
    numeric_columns = ['balls', 'strikes', 'outs_when_up', 'inning', 'at_bat_number']
    for col in numeric_columns:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # Ensure date is properly formatted
    data['game_date'] = pd.to_datetime(data['game_date'])
    
    return data

def create_game_situation_features(data):
    """
    Create features related to the game situation
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Cleaned pitch data
    
    Returns:
    --------
    pandas.DataFrame
        Data with added game situation features
    """
    # Count situation (hitter vs pitcher advantage)
    # For baseball, common hitter counts are 1-0, 2-0, 3-0, 2-1, 3-1, 3-2
    # Common pitcher counts are 0-1, 0-2, 1-2, 2-2
    
    # Create a count indicator
    data['count'] = data['balls'].astype(str) + '-' + data['strikes'].astype(str)
    
    # Flag for hitter vs pitcher counts
    hitter_counts = ['1-0', '2-0', '3-0', '2-1', '3-1', '3-2']
    pitcher_counts = ['0-1', '0-2', '1-2', '2-2']
    
    data['hitter_count'] = data['count'].isin(hitter_counts).astype(int)
    data['pitcher_count'] = data['count'].isin(pitcher_counts).astype(int)
    data['neutral_count'] = (~data['count'].isin(hitter_counts + pitcher_counts)).astype(int)
    
    # Early/late inning indicator
    data['early_inning'] = (data['inning'] <= 3).astype(int)
    data['middle_inning'] = ((data['inning'] > 3) & (data['inning'] <= 6)).astype(int)
    data['late_inning'] = (data['inning'] > 6).astype(int)
    
    # Pressure situation features
    # If runner on base and it's a close game (within 3 runs)
    if 'on_1b' in data.columns and 'on_2b' in data.columns and 'on_3b' in data.columns and 'home_score' in data.columns and 'away_score' in data.columns:
        data['runners_on'] = ((data['on_1b'].notna()) | (data['on_2b'].notna()) | (data['on_3b'].notna())).astype(int)
        data['score_diff'] = (data['home_score'] - data['away_score']).abs()
        data['close_game'] = (data['score_diff'] <= 3).astype(int)
        data['pressure_situation'] = (data['runners_on'] & data['close_game']).astype(int)
    
    return data

def calculate_pitcher_tendencies(data):
    """
    Calculate various pitcher tendencies as features
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Pitch data
    
    Returns:
    --------
    pandas.DataFrame
        Data with added pitcher tendency features
    """
    logger.info("Calculating pitcher tendencies")
    
    # Overall fastball percentage for each pitcher
    pitcher_fb_pct = data.groupby('pitcher')['is_fastball'].mean().reset_index()
    pitcher_fb_pct.columns = ['pitcher', 'pitcher_fb_pct']
    
    # Fastball percentage by count for each pitcher
    pitcher_count_fb = data.groupby(['pitcher', 'count'])['is_fastball'].mean().reset_index()
    pitcher_count_fb.columns = ['pitcher', 'count', 'pitcher_count_fb_pct']
    
    # Fastball percentage in hitter vs pitcher counts
    pitcher_hitter_count_fb = data.groupby(['pitcher', 'hitter_count'])['is_fastball'].mean().reset_index()
    pitcher_hitter_count_fb.columns = ['pitcher', 'hitter_count', 'pitcher_hitter_count_fb_pct']
    
    # Merge these back to the main dataframe
    data = pd.merge(data, pitcher_fb_pct, on='pitcher', how='left')
    data = pd.merge(data, pitcher_count_fb, on=['pitcher', 'count'], how='left')
    data = pd.merge(data, pitcher_hitter_count_fb, on=['pitcher', 'hitter_count'], how='left')
    
    return data

def calculate_catcher_tendencies(data):
    """
    Calculate catcher tendencies for pitch calling
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Pitch data
    
    Returns:
    --------
    pandas.DataFrame
        Data with added catcher tendency features
    """
    # Only proceed if we have catcher information
    if 'fielder_2' not in data.columns:
        logger.warning("No catcher information (fielder_2) found in the data")
        return data
    
    logger.info("Calculating catcher tendencies")
    
    # Overall fastball percentage for each catcher
    catcher_fb_pct = data.groupby('fielder_2')['is_fastball'].mean().reset_index()
    catcher_fb_pct.columns = ['fielder_2', 'catcher_fb_pct']
    
    # Fastball percentage by count for each catcher
    catcher_count_fb = data.groupby(['fielder_2', 'count'])['is_fastball'].mean().reset_index()
    catcher_count_fb.columns = ['fielder_2', 'count', 'catcher_count_fb_pct']
    
    # Merge these back to the main dataframe
    data = pd.merge(data, catcher_fb_pct, on='fielder_2', how='left')
    data = pd.merge(data, catcher_count_fb, on=['fielder_2', 'count'], how='left')
    
    return data

def calculate_batter_tendencies(data):
    """
    Calculate batter tendencies against different pitch types
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Pitch data
    
    Returns:
    --------
    pandas.DataFrame
        Data with added batter tendency features
    """
    logger.info("Calculating batter tendencies")
    
    # Calculate how well batters do against fastballs vs offspeed
    # We'll use events that ended in hits as a proxy for "success"
    
    if 'events' not in data.columns or 'description' not in data.columns:
        logger.warning("No event or description information found in the data")
        return data
    
    # Define hit events
    hit_events = ['single', 'double', 'triple', 'home_run']
    
    # Flag for contact success
    data['hit'] = data['events'].isin(hit_events).astype(int)
    
    # Calculate batter success rate against fastballs and offspeed
    fastball_data = data[data['is_fastball'] == 1]
    offspeed_data = data[data['is_fastball'] == 0]
    
    # Success against fastballs
    batter_vs_fb = fastball_data.groupby('batter')['hit'].mean().reset_index()
    batter_vs_fb.columns = ['batter', 'batter_success_vs_fb']
    
    # Success against offspeed
    batter_vs_os = offspeed_data.groupby('batter')['hit'].mean().reset_index()
    batter_vs_os.columns = ['batter', 'batter_success_vs_os']
    
    # Merge these back to the main dataframe
    data = pd.merge(data, batter_vs_fb, on='batter', how='left')
    data = pd.merge(data, batter_vs_os, on='batter', how='left')
    
    # Calculate the differential (whether batter is better against FB or OS)
    data['batter_fb_os_diff'] = data['batter_success_vs_fb'] - data['batter_success_vs_os']
    
    return data

def create_sequence_features(data):
    """
    Create features based on pitch sequencing
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Pitch data
    
    Returns:
    --------
    pandas.DataFrame
        Data with added sequence features
    """
    logger.info("Creating pitch sequence features")
    
    # Sort by game, at_bat_number, and pitch_number to ensure correct sequence
    if 'game_pk' in data.columns and 'at_bat_number' in data.columns and 'pitch_number' in data.columns:
        data = data.sort_values(by=['game_pk', 'at_bat_number', 'pitch_number'])
        
        # Create previous pitch type feature
        data['prev_pitch_type'] = data.groupby(['game_pk', 'at_bat_number'])['pitch_type'].shift(1)
        data['prev_is_fastball'] = data.groupby(['game_pk', 'at_bat_number'])['is_fastball'].shift(1)
        
        # Calculate the number of consecutive similar pitches
        data['consecutive_same_type'] = (data['pitch_type'] == data['prev_pitch_type']).astype(int)
        data['consecutive_counter'] = data.groupby(['game_pk', 'at_bat_number']).apply(
            lambda x: x['consecutive_same_type'].cumsum()
        ).reset_index(level=[0, 1], drop=True)
        
        # First pitch of at-bat indicator
        data['first_pitch'] = (data['pitch_number'] == 1).astype(int)
    else:
        logger.warning("Missing required columns for sequence features")
    
    return data

def prepare_modeling_data(data):
    """
    Prepare final dataset for modeling by handling missing values,
    encoding categorical variables, etc.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Processed pitch data
    
    Returns:
    --------
    pandas.DataFrame
        Final modeling dataset
    """
    logger.info("Preparing final modeling dataset")
    
    # Select relevant features for modeling
    model_features = [
        # Target variable
        'is_fastball',
        
        # Game situation
        'balls', 'strikes', 'outs_when_up', 'inning',
        'hitter_count', 'pitcher_count', 'neutral_count',
        'early_inning', 'middle_inning', 'late_inning',
        
        # Pitcher tendencies
        'pitcher_fb_pct', 'pitcher_count_fb_pct', 'pitcher_hitter_count_fb_pct',
        
        # Catcher tendencies (if available)
        'catcher_fb_pct', 'catcher_count_fb_pct',
        
        # Batter tendencies
        'batter_success_vs_fb', 'batter_success_vs_os', 'batter_fb_os_diff',
        
        # Pitch sequence
        'first_pitch', 'prev_is_fastball', 'consecutive_counter'
    ]
    
    # Only include columns that exist in the data
    available_features = [col for col in model_features if col in data.columns]
    model_data = data[available_features].copy()
    
    # Fill missing values with appropriate strategies
    # For most numeric features, median is a reasonable default
    for col in model_data.columns:
        if model_data[col].dtype in [np.float64, np.int64]:
            model_data[col] = model_data[col].fillna(model_data[col].median())
    
    logger.info(f"Final modeling dataset shape: {model_data.shape}")
    return model_data

def main():
    """Main function to execute data preprocessing workflow"""
    # Create directories if they don't exist
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    # Find the most recent classified pitch data file
    classified_files = glob.glob(os.path.join(PROCESSED_DATA_DIR, "pitches_classified_*.csv"))
    
    if not classified_files:
        logger.error("No classified pitch data found. Run data_collection.py first.")
        return
    
    # Sort by modification time to get the most recent
    most_recent_file = max(classified_files, key=os.path.getmtime)
    logger.info(f"Processing most recent file: {most_recent_file}")
    
    # Load classified pitch data
    pitch_data = load_data(most_recent_file)
    
    # Clean the data
    cleaned_data = clean_data(pitch_data)
    
    # Create game situation features
    data_with_situation = create_game_situation_features(cleaned_data)
    
    # Add pitcher tendencies
    data_with_pitcher = calculate_pitcher_tendencies(data_with_situation)
    
    # Add catcher tendencies
    data_with_catcher = calculate_catcher_tendencies(data_with_pitcher)
    
    # Add batter tendencies
    data_with_batter = calculate_batter_tendencies(data_with_catcher)
    
    # Add sequence features
    data_with_sequence = create_sequence_features(data_with_batter)
    
    # Prepare final modeling dataset
    modeling_data = prepare_modeling_data(data_with_sequence)
    
    # Save the modeling-ready dataset
    # Get the date range from the input file name for consistency
    file_name_parts = os.path.basename(most_recent_file).split('_')
    date_info = '_'.join(file_name_parts[2:]) if len(file_name_parts) > 2 else 'unknown_dates'
    
    output_path = os.path.join(PROCESSED_DATA_DIR, f"modeling_data_{date_info}")
    modeling_data.to_csv(output_path, index=False)
    logger.info(f"Modeling-ready data saved to {output_path}")
    
    # Output some statistics
    logger.info(f"Total samples: {len(modeling_data)}")
    logger.info(f"Fastball percentage: {modeling_data['is_fastball'].mean() * 100:.2f}%")
    logger.info(f"Number of features: {len(modeling_data.columns) - 1}")  # Excluding target

if __name__ == "__main__":
    main() 