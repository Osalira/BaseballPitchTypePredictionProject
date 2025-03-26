#!/usr/bin/env python
"""
Baseball Pitch Prediction Pipeline

This script runs the entire pipeline from data collection to modeling
and prediction.
"""

import os
import sys
import argparse
import subprocess
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_script(script_path, description, args=None):
    """
    Run a Python script and log the output
    
    Parameters:
    -----------
    script_path : str
        Path to the Python script
    description : str
        Description of the step for logging
    args : list, optional
        Command line arguments to pass to the script
    
    Returns:
    --------
    int
        Return code from the script (0 if successful)
    """
    logger.info(f"Starting: {description}")
    start_time = time.time()
    
    try:
        # Build the command
        command = [sys.executable, script_path]
        if args:
            command.extend(args)
            
        # Run the script as a subprocess
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True
        )

        # Log the output
        if result.stdout:
            for line in result.stdout.split('\n'):
                if line.strip():
                    logger.info(f"[{os.path.basename(script_path)}] {line}")
        
        # Log any errors
        if result.stderr:
            for line in result.stderr.split('\n'):
                if line.strip():
                    logger.warning(f"[{os.path.basename(script_path)}] {line}")
        
        elapsed_time = time.time() - start_time
        logger.info(f"Completed: {description} (Time: {elapsed_time:.2f}s)")
        
        return result.returncode
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Error in {description}: {str(e)}")
        if e.stdout:
            for line in e.stdout.split('\n'):
                if line.strip():
                    logger.info(f"[{os.path.basename(script_path)}] {line}")
        if e.stderr:
            for line in e.stderr.split('\n'):
                if line.strip():
                    logger.error(f"[{os.path.basename(script_path)}] {line}")
        
        return e.returncode
    
    except Exception as e:
        logger.error(f"Failed to run {description}: {str(e)}")
        return 1

def main():
    """Main function to run the pipeline"""
    parser = argparse.ArgumentParser(description='Run Baseball Pitch Prediction Pipeline')
    
    # Date range options for single-season collection
    parser.add_argument('--start-date', type=str, help='Start date for data collection (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date for data collection (YYYY-MM-DD)')
    
    # Multi-season collection options
    parser.add_argument('--seasons', nargs='+', 
                      help='Specific seasons to collect (e.g., 2021 2022 2023)')
    parser.add_argument('--skip-existing', action='store_true',
                      help='Skip seasons that already have data files')
    
    # Pipeline control options
    parser.add_argument('--skip-collection', action='store_true', help='Skip data collection step')
    parser.add_argument('--skip-preprocessing', action='store_true', help='Skip data preprocessing step')
    parser.add_argument('--skip-training', action='store_true', help='Skip model training step')
    parser.add_argument('--interactive', action='store_true', help='Run interactive prediction at the end')
    
    args = parser.parse_args()
    
    # Create directories if they don't exist
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('models/results', exist_ok=True)
    
    # Step 1: Data Collection
    if not args.skip_collection:
        # Determine collection method
        if args.seasons:
            # Use multi-season collection
            logger.info(f"Collecting data for multiple seasons: {', '.join(args.seasons)}")
            collect_args = ['--seasons'] + args.seasons
            if args.skip_existing:
                collect_args.append('--skip-existing')
            
            result = run_script('src/collect_seasons.py', "Multi-Season Data Collection", collect_args)
            if result != 0:
                logger.error("Multi-season data collection failed, stopping pipeline")
                return result
        
        elif args.start_date and args.end_date:
            # Use custom date range for single-season collection
            logger.info(f"Setting custom date range: {args.start_date} to {args.end_date}")
            with open('src/data_collection.py', 'r') as f:
                content = f.read()
            
            # Replace default date range with custom dates
            content = content.replace(
                "start_date = '2022-04-07'",
                f"start_date = '{args.start_date}'"
            )
            content = content.replace(
                "end_date = '2022-10-05'",
                f"end_date = '{args.end_date}'"
            )
            
            with open('src/data_collection_custom.py', 'w') as f:
                f.write(content)
            
            # Run custom data collection
            result = run_script('src/data_collection_custom.py', "Custom Date Range Data Collection")
            if result != 0:
                logger.error("Data collection failed, stopping pipeline")
                return result
        
        else:
            # Use default season collection
            result = run_script('src/data_collection.py', "Data Collection")
            if result != 0:
                logger.error("Data collection failed, stopping pipeline")
                return result
    else:
        logger.info("Skipping data collection step")
    
    # Step 2: Data Preprocessing
    if not args.skip_preprocessing:
        result = run_script('src/data_preprocessing.py', "Data Preprocessing")
        if result != 0:
            logger.error("Data preprocessing failed, stopping pipeline")
            return result
    else:
        logger.info("Skipping data preprocessing step")
    
    # Step 3: Model Training
    if not args.skip_training:
        result = run_script('src/model_training.py', "Model Training")
        if result != 0:
            logger.error("Model training failed, stopping pipeline")
            return result
    else:
        logger.info("Skipping model training step")
    
    # Step 4: Run interactive prediction if requested
    if args.interactive:
        logger.info("Running interactive prediction")
        command = [sys.executable, "src/predict.py", "--interactive"]
        subprocess.run(command)
    
    logger.info("Pipeline completed successfully!")
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code) 