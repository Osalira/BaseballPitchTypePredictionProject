"""
Flexible MLB Season Collection Script

This script allows collection of MLB Statcast data for multiple seasons to enhance
the baseball pitch prediction model with more training data.
"""

import os
import pandas as pd
import argparse
import logging
import sys
import glob
from datetime import datetime

# Import from existing code
sys.path.append('src')
from data_collection import collect_data_in_chunks, categorize_pitches

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("seasons_collection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Directory paths
RAW_DATA_DIR = os.path.join('data', 'raw')
PROCESSED_DATA_DIR = os.path.join('data', 'processed')

# Common MLB season date ranges
SEASON_DATES = {
    "2019": {"start": "2019-03-28", "end": "2019-09-29"},
    "2020": {"start": "2020-07-23", "end": "2020-09-27"},  # COVID-shortened season
    "2021": {"start": "2021-04-01", "end": "2021-10-03"},
    "2022": {"start": "2022-04-07", "end": "2022-10-05"},
    "2023": {"start": "2023-03-30", "end": "2023-10-01"},
    "2024": {"start": "2024-03-28", "end": "2024-09-29"}
}

def collect_season_data(season, skip_existing=False):
    """
    Collect data for a specific MLB season
    
    Parameters:
    -----------
    season : str
        The season year to collect (e.g., "2022")
    skip_existing : bool
        Whether to skip collection if season data already exists
        
    Returns:
    --------
    pandas.DataFrame
        The collected season data
    """
    season_file = os.path.join(RAW_DATA_DIR, f"statcast_season_{season}.csv")
    
    # Skip if file exists and skip_existing flag is set
    if os.path.exists(season_file) and skip_existing:
        logger.info(f"Season {season} data already exists. Loading from file.")
        season_data = pd.read_csv(season_file)
        return season_data
    
    # Collect the data
    dates = SEASON_DATES.get(season)
    if not dates:
        logger.error(f"No date range defined for season {season}")
        raise ValueError(f"Unknown season: {season}")
    
    logger.info(f"Collecting data for season {season} from {dates['start']} to {dates['end']}")
    
    # Initialize list to store successful chunks
    successful_chunks = []
    
    try:
        # Collect data in chunks
        season_data = collect_data_in_chunks(dates['start'], dates['end'], chunk_days=14)
        successful_chunks.append(season_data)
    except Exception as e:
        logger.error(f"Error during collection of season {season}: {str(e)}")
        # Look for any partially collected chunks
        chunk_pattern = os.path.join(RAW_DATA_DIR, f"statcast_*_{season}-*.csv")
        chunk_files = glob.glob(chunk_pattern)
        
        if chunk_files:
            logger.info(f"Found {len(chunk_files)} partial chunks for season {season}")
            for chunk_file in chunk_files:
                try:
                    chunk_data = pd.read_csv(chunk_file)
                    successful_chunks.append(chunk_data)
                    logger.info(f"Loaded partial chunk from {chunk_file}")
                except Exception as chunk_e:
                    logger.error(f"Error loading chunk {chunk_file}: {str(chunk_e)}")
    
    if not successful_chunks:
        raise ValueError(f"No data could be collected for season {season}")
    
    # Combine all successful chunks
    season_data = pd.concat(successful_chunks, ignore_index=True)
    
    # Save combined season data
    season_data.to_csv(season_file, index=False)
    logger.info(f"Season {season} data saved to {season_file} ({len(season_data)} records)")
    
    return season_data

def add_season_features(data):
    """
    Add season-related features to the dataset
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The pitch data
        
    Returns:
    --------
    pandas.DataFrame
        Data with added season features
    """
    # Convert game_date to datetime if not already
    data['game_date'] = pd.to_datetime(data['game_date'])
    
    # Extract season (year)
    data['season'] = data['game_date'].dt.year
    
    # Create season indicator variables (one-hot encoding)
    seasons = data['season'].unique()
    for season in seasons:
        data[f'season_{season}'] = (data['season'] == season).astype(int)
    
    return data

def main():
    """Main function to execute the multi-season collection workflow"""
    parser = argparse.ArgumentParser(description='Collect MLB data for specific seasons')
    parser.add_argument('--seasons', nargs='+', choices=SEASON_DATES.keys(), required=True,
                      help='List of seasons to collect (e.g., --seasons 2021 2022 2023)')
    parser.add_argument('--skip-existing', action='store_true', 
                      help='Skip seasons that already have data files')
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    all_season_data = []
    collected_seasons = []
    
    # Collect each season
    for season in args.seasons:
        try:
            season_data = collect_season_data(season, args.skip_existing)
            all_season_data.append(season_data)
            collected_seasons.append(season)
            logger.info(f"Successfully collected {len(season_data)} pitches for season {season}")
        except Exception as e:
            logger.error(f"Error collecting season {season}: {str(e)}")
            if not args.skip_existing:
                logger.error("Stopping collection due to error. Use --skip-existing to continue with other seasons.")
                return
    
    if not collected_seasons:
        logger.error("No seasons were collected. Nothing to process.")
        return
    
    # Combine seasons
    season_range = f"{min(collected_seasons)}_to_{max(collected_seasons)}"
    combined_data = pd.concat(all_season_data, ignore_index=True)
    
    # Save combined raw data
    combined_path = os.path.join(RAW_DATA_DIR, f"statcast_combined_{season_range}.csv")
    combined_data.to_csv(combined_path, index=False)
    logger.info(f"Combined data saved to {combined_path}")
    
    # Process and categorize pitches
    processed_data = categorize_pitches(combined_data)
    
    # Add season features
    processed_data = add_season_features(processed_data)
    
    # Save processed data
    processed_path = os.path.join(PROCESSED_DATA_DIR, f"pitches_classified_{season_range}.csv")
    processed_data.to_csv(processed_path, index=False)
    logger.info(f"Processed data saved to {processed_path}")
    
    # Print stats
    logger.info(f"Total pitches collected: {len(processed_data)}")
    for season in collected_seasons:
        season_count = len(processed_data[processed_data['season'] == int(season)])
        logger.info(f"Season {season}: {season_count} pitches")
        
    logger.info(f"Fastball percentage: {processed_data['is_fastball'].mean() * 100:.2f}%")
    
    # Print next steps
    logger.info("\nNext steps:")
    logger.info("1. Run 'python run_pipeline.py --skip-collection' to preprocess and train models with this data")
    logger.info("2. Check the notebooks for exploratory data analysis and model evaluation")

if __name__ == "__main__":
    main() 