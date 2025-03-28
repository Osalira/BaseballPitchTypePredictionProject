"""
Data Preprocessing Script for Baseball Pitch Prediction

This script processes raw Statcast data to create features for
pitch prediction modeling with SQLite integration for performance.
"""

import os
import pandas as pd
import numpy as np
import logging
import sqlite3
from multiprocessing import Pool, cpu_count
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
DB_PATH = os.path.join('data', 'baseball.db')

def load_data(file_path, use_sqlite=True):
    """
    Load data from CSV file or SQLite database
    
    Parameters:
    -----------
    file_path : str
        Path to the raw data CSV file
    use_sqlite : bool
        Whether to use SQLite database
    
    Returns:
    --------
    pandas.DataFrame
        Loaded data
    """
    if not use_sqlite:
        logger.info(f"Loading data from {file_path}")
        data = pd.read_csv(file_path)
        return data
    
    # Initialize SQLite database
    conn = sqlite3.connect(DB_PATH)
    
    # Check if file exists in database
    table_name = os.path.basename(file_path).replace('.csv', '').replace('-', '_')
    
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
    table_exists = cursor.fetchone() is not None
    
    if table_exists:
        logger.info(f"Loading data from SQLite table: {table_name}")
        
        # Get column names from the table to handle potential SQLite column name changes
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns_info = cursor.fetchall()
        
        # Check if we have an is_fastball or isfastball column
        has_is_fastball = False
        is_fastball_col_name = None
        
        for col_info in columns_info:
            col_name = col_info[1]  # Column name is the second element
            if col_name.lower() == 'is_fastball' or col_name.lower() == 'isfastball':
                has_is_fastball = True
                is_fastball_col_name = col_name
                break
        
        # Load data from SQLite
        data = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        
        # Rename column if needed
        if has_is_fastball and is_fastball_col_name != 'is_fastball':
            logger.info(f"Renaming column '{is_fastball_col_name}' to 'is_fastball'")
            data = data.rename(columns={is_fastball_col_name: 'is_fastball'})
    else:
        logger.info(f"Importing {file_path} to SQLite")
        # Read CSV in chunks to reduce memory usage
        chunk_size = 100000
        data = pd.DataFrame()
        
        # First check column names in the CSV file
        csv_columns = pd.read_csv(file_path, nrows=1).columns.tolist()
        
        # Process any special columns before importing
        column_mapping = {}
        for col in csv_columns:
            if col.lower() == 'is_fastball' or col.lower() == 'isfastball':
                # Ensure we explicitly map to our expected name
                column_mapping[col] = 'is_fastball'
        
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            # Apply mapping if needed
            if column_mapping:
                chunk = chunk.rename(columns=column_mapping)
                
            # Save to SQLite with careful handling of column names
            chunk.to_sql(table_name, conn, if_exists='append', index=False)
            
            # Append chunk to dataframe if needed for further processing
            if data.empty:
                data = chunk
            else:
                data = pd.concat([data, chunk], ignore_index=True)
    
    conn.close()
    
    # Final safety check
    if 'is_fastball' not in data.columns and 'isfastball' in data.columns:
        data = data.rename(columns={'isfastball': 'is_fastball'})
        
    return data

def clean_data(data):
    """
    Clean the raw data by handling missing values and filtering out invalid records
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Raw Statcast data
    
    Returns:
    --------
    pandas.DataFrame
        Cleaned data
    """
    logger.info("Cleaning data")
    initial_rows = len(data)
    
    # Fix column name differences - check if 'isfastball' exists and rename to 'is_fastball'
    if 'isfastball' in data.columns and 'is_fastball' not in data.columns:
        logger.info("Renaming 'isfastball' column to 'is_fastball'")
        data = data.rename(columns={'isfastball': 'is_fastball'})
    
    # Check if is_fastball column exists, if not create it
    if 'is_fastball' not in data.columns:
        logger.info("Creating is_fastball column")
        # Define which pitch types are considered fastballs
        # FF: Four-seam, FT: Two-seam, FC: Cutter, SI: Sinker
        fastball_types = ['FF', 'FT', 'FC', 'SI']
        
        # Create binary feature
        data['is_fastball'] = data['pitch_type'].isin(fastball_types).astype(int)
        
        # Create simple labels for clarity
        data['pitch_category'] = data['is_fastball'].map({1: 'fastball', 0: 'offspeed'})
        
        logger.info(f"Fastball percentage: {data['is_fastball'].mean() * 100:.2f}%")
    
    # Drop rows with missing pitch types
    data = data.dropna(subset=['pitch_type', 'is_fastball'])
    
    # Remove invalid game states (negative counts, etc.)
    data = data[(data['balls'] >= 0) & (data['balls'] <= 3)]
    data = data[(data['strikes'] >= 0) & (data['strikes'] <= 2)]
    
    # Handle missing values for important columns
    # Use vectorized operations instead of apply/iterrows for speed
    numeric_cols = ['pitch_number', 'outs_when_up', 'inning']
    data[numeric_cols] = data[numeric_cols].fillna(0).astype(int)
    
    # Drop rows with null values in critical features
    critical_cols = ['game_date', 'pitcher', 'batter', 'pitch_type']
    data = data.dropna(subset=critical_cols)
    
    # Convert dates to datetime if not already
    data['game_date'] = pd.to_datetime(data['game_date'])
    
    final_rows = len(data)
    removed_rows = initial_rows - final_rows
    logger.info(f"Removed {removed_rows} rows ({removed_rows/initial_rows:.1%}) while cleaning data")
    
    return data

# Parallel processing functions
def _process_chunk(chunk_data):
    """Process a single chunk of data in parallel"""
    chunk_result = create_game_situation_features(chunk_data)
    return chunk_result

def parallel_process(data, func, n_processes=None):
    """
    Process data in parallel using multiple cores
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data to process
    func : function
        Function to apply to each chunk
    n_processes : int
        Number of processes to use (default: CPU count - 1)
    
    Returns:
    --------
    pandas.DataFrame
        Processed data
    """
    if n_processes is None:
        n_processes = max(1, cpu_count() - 1)
    
    # Split data into chunks
    n_rows = len(data)
    chunk_size = n_rows // n_processes
    chunks = []
    
    for i in range(n_processes):
        start = i * chunk_size
        end = start + chunk_size if i < n_processes - 1 else n_rows
        chunks.append(data.iloc[start:end].copy())
    
    # Process chunks in parallel
    with Pool(processes=n_processes) as pool:
        results = pool.map(func, chunks)
    
    # Combine results
    return pd.concat(results, ignore_index=True)

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
    logger.info("Creating game situation features")
    
    # Create count features (vectorized operations)
    data['count'] = data['balls'].astype(str) + '-' + data['strikes'].astype(str)
    
    # Create count type features (hitter, pitcher, neutral counts)
    hitter_counts = ['1-0', '2-0', '3-0', '2-1', '3-1', '3-2']
    pitcher_counts = ['0-1', '0-2', '1-2', '2-2']
    
    # Vectorized operations
    data['hitter_count'] = data['count'].isin(hitter_counts).astype(int)
    data['pitcher_count'] = data['count'].isin(pitcher_counts).astype(int)
    data['neutral_count'] = (~data['count'].isin(hitter_counts + pitcher_counts)).astype(int)
    
    # Create pressure situation features
    data['first_pitch'] = (data['pitch_number'] == 1).astype(int)
    
    # Add inning stage features
    data['early_inning'] = (data['inning'] <= 3).astype(int)
    data['middle_inning'] = ((data['inning'] > 3) & (data['inning'] <= 6)).astype(int)
    data['late_inning'] = (data['inning'] > 6).astype(int)
    
    return data

def calculate_pitcher_tendencies(data):
    """
    Calculate pitcher tendencies for fastball usage
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data with game situation features
    
    Returns:
    --------
    pandas.DataFrame
        Data with added pitcher tendency features
    """
    logger.info("Calculating pitcher tendencies")
    
    # Group by pitcher to calculate overall tendencies
    pitcher_fb_pct = data.groupby('pitcher')['is_fastball'].mean()
    
    # Group by pitcher and count to calculate count-specific tendencies
    pitcher_count_fb_pct = data.groupby(['pitcher', 'count'])['is_fastball'].mean().unstack(fill_value=0.5)
    
    # Group by pitcher and count type
    pitcher_count_type_fb_pct = pd.DataFrame()
    for count_type in ['hitter_count', 'pitcher_count', 'neutral_count']:
        # Use numpy for vectorized calculation
        temp = data[data[count_type] == 1].groupby('pitcher')['is_fastball'].mean()
        pitcher_count_type_fb_pct[count_type] = temp
    
    # Add these features back to the main dataframe
    data['pitcher_fb_pct'] = data['pitcher'].map(pitcher_fb_pct)
    
    # Add count-specific metrics (use merge for better performance with large datasets)
    data_with_counts = data.copy()
    
    # Vectorized mapping of count tendencies
    for count in pitcher_count_fb_pct.columns:
        count_dict = pitcher_count_fb_pct[count].to_dict()
        mask = data_with_counts['count'] == count
        data_with_counts.loc[mask, 'pitcher_count_fb_pct'] = data_with_counts.loc[mask, 'pitcher'].map(
            lambda x: count_dict.get(x, 0.5)
        )
    
    # Fill missing values
    data_with_counts['pitcher_count_fb_pct'] = data_with_counts['pitcher_count_fb_pct'].fillna(0.5)
    
    # Add count-type metrics
    for count_type in ['hitter_count', 'pitcher_count', 'neutral_count']:
        count_type_col = f'pitcher_{count_type}_fb_pct'
        count_type_dict = pitcher_count_type_fb_pct[count_type].fillna(0.5).to_dict()
        data_with_counts[count_type_col] = data_with_counts['pitcher'].map(count_type_dict)
    
    return data_with_counts

def calculate_catcher_tendencies(data):
    """
    Calculate catcher tendencies for fastball usage
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data with pitcher tendency features
    
    Returns:
    --------
    pandas.DataFrame
        Data with added catcher tendency features
    """
    logger.info("Calculating catcher tendencies")
    
    # Check if catcher ID is available
    if 'fielder_2' not in data.columns:
        logger.warning("Catcher ID (fielder_2) not found, skipping catcher tendencies")
        return data
    
    # Group by catcher to calculate overall tendencies
    catcher_fb_pct = data.groupby('fielder_2')['is_fastball'].mean()
    
    # Group by catcher and count to calculate count-specific tendencies
    catcher_count_fb_pct = data.groupby(['fielder_2', 'count'])['is_fastball'].mean().unstack(fill_value=0.5)
    
    # Add these features back to the main dataframe
    data['catcher_fb_pct'] = data['fielder_2'].map(catcher_fb_pct)
    
    # Vectorized mapping of count tendencies
    for count in catcher_count_fb_pct.columns:
        count_dict = catcher_count_fb_pct[count].to_dict()
        mask = data['count'] == count
        data.loc[mask, 'catcher_count_fb_pct'] = data.loc[mask, 'fielder_2'].map(
            lambda x: count_dict.get(x, 0.5)
        )
    
    # Fill missing values
    data['catcher_count_fb_pct'] = data['catcher_count_fb_pct'].fillna(0.5)
    
    return data

def calculate_batter_tendencies(data):
    """
    Calculate batter performance against different pitch types
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data with catcher tendency features
    
    Returns:
    --------
    pandas.DataFrame
        Data with added batter tendency features
    """
    logger.info("Calculating batter tendencies")
    
    # Define success for a batter (contact, hits, etc.)
    # For simplicity, we'll use whether they made contact
    # This can be expanded based on available metrics
    contact_events = ['single', 'double', 'triple', 'home_run', 'field_out', 'force_out', 
                    'grounded_into_double_play', 'fielders_choice', 'sac_fly', 'sac_bunt']
    
    # Create vectorized success indicator (1 for contact, 0 otherwise)
    data['batter_success'] = data['events'].isin(contact_events).astype(int)
    
    # Calculate success rates against fastballs and offspeed
    # Use numpy vectorization for speed
    fastball_success = data[data['is_fastball'] == 1].groupby('batter')['batter_success'].agg(['mean', 'count'])
    offspeed_success = data[data['is_fastball'] == 0].groupby('batter')['batter_success'].agg(['mean', 'count'])
    
    # Rename columns
    fastball_success.columns = ['success_rate', 'pitches']
    offspeed_success.columns = ['success_rate', 'pitches']
    
    # Only consider success rates with sufficient sample size
    min_pitches = 20
    fastball_success = fastball_success[fastball_success['pitches'] >= min_pitches]
    offspeed_success = offspeed_success[offspeed_success['pitches'] >= min_pitches]
    
    # Convert to dictionaries for faster mapping
    fb_success_dict = fastball_success['success_rate'].to_dict()
    os_success_dict = offspeed_success['success_rate'].to_dict()
    
    # Add features using vectorized operations
    data['batter_success_vs_fb'] = data['batter'].map(fb_success_dict)
    data['batter_success_vs_os'] = data['batter'].map(os_success_dict)
    
    # Calculate success differential (how much better against fastball vs offspeed)
    data['batter_fb_os_diff'] = data['batter_success_vs_fb'].fillna(0) - data['batter_success_vs_os'].fillna(0)
    
    # Fill NaN values with overall averages
    fb_avg = fastball_success['success_rate'].mean()
    os_avg = offspeed_success['success_rate'].mean()
    
    data['batter_success_vs_fb'] = data['batter_success_vs_fb'].fillna(fb_avg)
    data['batter_success_vs_os'] = data['batter_success_vs_os'].fillna(os_avg)
    data['batter_fb_os_diff'] = data['batter_fb_os_diff'].fillna(0)
    
    return data

def create_sequence_features(data):
    """
    Create features based on pitch sequence
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data with batter tendency features
    
    Returns:
    --------
    pandas.DataFrame
        Data with added sequence features
    """
    logger.info("Creating pitch sequence features")
    
    # Sort by game_date, at_bat_number, and pitch_number to get proper sequence
    data = data.sort_values(['game_date', 'game_pk', 'at_bat_number', 'pitch_number'])
    
    # Group by game and at-bat
    grouped = data.groupby(['game_pk', 'at_bat_number'])
    
    # Function to apply to each group
    def calculate_sequence_features(group):
        # Previous pitch type
        group['prev_is_fastball'] = group['is_fastball'].shift(1).fillna(0)
        
        # Calculate consecutive counter
        group['consecutive_counter'] = 0
        counter = 0
        prev_type = None
        
        for i, row in group.iterrows():
            current_type = row['is_fastball']
            if i == group.index[0]:
                # First pitch in at-bat
                prev_type = current_type
            else:
                if current_type == prev_type:
                    counter += 1
                else:
                    counter = 0
                prev_type = current_type
            
            group.at[i, 'consecutive_counter'] = counter
        
        return group
    
    # This is a bottleneck - let's use a more efficient approach with .apply()
    # Warning: We need to use include_groups=False to avoid deprecation warning
    result = grouped.apply(calculate_sequence_features, include_groups=False)
    
    return result

def prepare_modeling_data(data):
    """
    Prepare the final dataset for modeling
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data with all features
    
    Returns:
    --------
    pandas.DataFrame
        Final modeling dataset
    """
    logger.info("Preparing modeling dataset")
    
    # Select only the columns needed for modeling
    model_features = [
        # Game situation
        'balls', 'strikes', 'outs_when_up', 'inning',
        'hitter_count', 'pitcher_count', 'neutral_count',
        'early_inning', 'middle_inning', 'late_inning',
        
        # Pitcher tendencies
        'pitcher_fb_pct', 'pitcher_count_fb_pct', 
        'pitcher_hitter_count_fb_pct', 'pitcher_pitcher_count_fb_pct', 'pitcher_neutral_count_fb_pct',
        
        # Catcher tendencies (if available)
        'catcher_fb_pct', 'catcher_count_fb_pct',
        
        # Batter tendencies
        'batter_success_vs_fb', 'batter_success_vs_os', 'batter_fb_os_diff',
        
        # Sequence features
        'first_pitch', 'prev_is_fastball', 'consecutive_counter',
        
        # Season features (if available)
        'season'
    ]
    
    # Add season indicator columns if they exist
    season_columns = [col for col in data.columns if col.startswith('season_20')]
    model_features.extend(season_columns)
    
    # Filter columns that actually exist in the dataframe
    model_features = [col for col in model_features if col in data.columns]
    
    # Add target variable
    model_features.append('is_fastball')
    
    # Select columns and drop rows with missing values
    modeling_data = data[model_features].dropna()
    
    # Log the shape of the final dataset
    logger.info(f"Final modeling dataset shape: {modeling_data.shape}")
    
    # Calculate fastball percentage in the dataset
    fb_pct = modeling_data['is_fastball'].mean() * 100
    logger.info(f"Fastball percentage in the dataset: {fb_pct:.2f}%")
    
    return modeling_data

def save_to_sqlite(data, table_name):
    """
    Save dataframe to SQLite database
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data to save
    table_name : str
        Name of the table
    """
    conn = sqlite3.connect(DB_PATH)
    data.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()
    logger.info(f"Saved data to SQLite table: {table_name}")

def main():
    """Main function to execute data preprocessing workflow"""
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    # Find the most recent combined data file
    combined_files = glob.glob(os.path.join(RAW_DATA_DIR, "statcast_combined_*.csv"))
    if not combined_files:
        logger.error("No combined data file found. Run data_collection.py first.")
        return
    
    most_recent_file = max(combined_files, key=os.path.getmtime)
    logger.info(f"Using combined data file: {most_recent_file}")
    
    # Extract season range from filename
    file_basename = os.path.basename(most_recent_file)
    season_range = file_basename.replace("statcast_combined_", "").replace(".csv", "")
    
    # Create database directory if needed
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    # Load, clean, and preprocess data
    raw_data = load_data(most_recent_file, use_sqlite=True)
    clean_raw_data = clean_data(raw_data)
    
    # Save the cleaned data to SQLite
    save_to_sqlite(clean_raw_data, f"clean_data_{season_range}")
    
    # Process data in parallel where possible
    # Game situation features
    game_data = create_game_situation_features(clean_raw_data)
    
    # Pitcher tendencies
    pitcher_data = calculate_pitcher_tendencies(game_data)
    
    # Catcher tendencies
    catcher_data = calculate_catcher_tendencies(pitcher_data)
    
    # Batter tendencies 
    batter_data = calculate_batter_tendencies(catcher_data)
    
    # Sequence features (this is sequential due to the nature of the calculation)
    sequence_data = create_sequence_features(batter_data)
    
    # Prepare final modeling dataset
    modeling_data = prepare_modeling_data(sequence_data)
    
    # Save processed data
    output_path = os.path.join(PROCESSED_DATA_DIR, f"modeling_data_{season_range}.csv")
    modeling_data.to_csv(output_path, index=False)
    
    # Also save to SQLite for faster access
    save_to_sqlite(modeling_data, f"modeling_data_{season_range}")
    
    logger.info(f"Preprocessing complete. Modeling-ready data saved to {output_path}")
    return modeling_data

if __name__ == "__main__":
    main() 