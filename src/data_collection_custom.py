"""
Data Collection Script for Baseball Pitch Prediction

This script uses pybaseball to collect MLB Statcast data for pitches,
focusing on creating a dataset to predict fastball vs offspeed pitches.
"""

import os
import pandas as pd
from pybaseball import statcast
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_collection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Directory paths
RAW_DATA_DIR = os.path.join('data', 'raw')
PROCESSED_DATA_DIR = os.path.join('data', 'processed')

def collect_statcast_data(start_date, end_date, save_path):
    """
    Collect MLB Statcast data for the specified date range
    
    Parameters:
    -----------
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    save_path : str
        Path to save the raw data CSV file
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the Statcast data
    """
    logger.info(f"Collecting Statcast data from {start_date} to {end_date}")
    try:
        data = statcast(start_dt=start_date, end_dt=end_date)
        logger.info(f"Successfully collected {len(data)} pitch records")
        
        # Save raw data
        data.to_csv(save_path, index=False)
        logger.info(f"Data saved to {save_path}")
        
        return data
    except Exception as e:
        logger.error(f"Error collecting data: {str(e)}")
        raise

def collect_data_in_chunks(start_date, end_date, chunk_days=7):
    """
    Collect MLB Statcast data in smaller chunks to avoid timeout issues
    
    Parameters:
    -----------
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    chunk_days : int
        Number of days per chunk
    
    Returns:
    --------
    pandas.DataFrame
        Combined DataFrame with all collected data
    """
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    all_data = []
    current = start
    chunk_num = 1
    
    while current < end:
        chunk_end = min(current + timedelta(days=chunk_days), end)
        
        chunk_start_str = current.strftime('%Y-%m-%d')
        chunk_end_str = chunk_end.strftime('%Y-%m-%d')
        
        save_path = os.path.join(RAW_DATA_DIR, f"statcast_{chunk_start_str}_to_{chunk_end_str}.csv")
        
        logger.info(f"Processing chunk {chunk_num}: {chunk_start_str} to {chunk_end_str}")
        
        # Check if file already exists to avoid redownloading
        if os.path.exists(save_path):
            logger.info(f"File {save_path} already exists. Loading from file.")
            chunk_data = pd.read_csv(save_path)
        else:
            chunk_data = collect_statcast_data(chunk_start_str, chunk_end_str, save_path)
        
        all_data.append(chunk_data)
        current = chunk_end + timedelta(days=1)
        chunk_num += 1
    
    # Combine all chunks into one DataFrame
    combined_data = pd.concat(all_data, ignore_index=True)
    
    # Save combined data
    combined_path = os.path.join(RAW_DATA_DIR, f"statcast_combined_{start_date}_to_{end_date}.csv")
    combined_data.to_csv(combined_path, index=False)
    logger.info(f"Combined data saved to {combined_path}")
    
    return combined_data

def categorize_pitches(data):
    """
    Add binary classification column to identify fastball vs offspeed
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Raw Statcast data
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added 'is_fastball' column
    """
    # Define which pitch types are considered fastballs
    # FF: Four-seam, FT: Two-seam, FC: Cutter, SI: Sinker
    fastball_types = ['FF', 'FT', 'FC', 'SI']
    
    # Create binary feature
    data['is_fastball'] = data['pitch_type'].isin(fastball_types).astype(int)
    
    # Create simple labels for clarity
    data['pitch_category'] = data['is_fastball'].map({1: 'fastball', 0: 'offspeed'})
    
    return data

def main():
    """Main function to execute data collection process"""
    # Create directories if they don't exist
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    # Define date range for data collection (adjust as needed)
    # Example: Collect data from the 2022 MLB season
    start_date = '2021-04-01'  # Opening day 2022
    end_date = '2021-10-03'    # Last day of 2022 regular season
    
    # Collect raw data
    combined_data = collect_data_in_chunks(start_date, end_date, chunk_days=14)
    
    # Basic processing - add fastball classification
    processed_data = categorize_pitches(combined_data)
    
    # Save processed data
    processed_path = os.path.join(PROCESSED_DATA_DIR, f"pitches_classified_{start_date}_to_{end_date}.csv")
    processed_data.to_csv(processed_path, index=False)
    logger.info(f"Processed data with pitch classification saved to {processed_path}")
    
    # Output some basic statistics
    total_pitches = len(processed_data)
    fastball_count = processed_data['is_fastball'].sum()
    fastball_pct = (fastball_count / total_pitches) * 100
    
    logger.info(f"Total pitches collected: {total_pitches}")
    logger.info(f"Fastball count: {fastball_count} ({fastball_pct:.2f}%)")
    logger.info(f"Offspeed count: {total_pitches - fastball_count} ({100 - fastball_pct:.2f}%)")

if __name__ == "__main__":
    main() 