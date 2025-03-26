#!/usr/bin/env python
"""
Multi-Season Training Demo

This script demonstrates how to collect and train a model using multiple seasons of MLB data.
It serves as a quick example of how to use the multi-season features of the pipeline.
"""

import subprocess
import sys
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Run a complete multi-season training workflow"""
    logger.info("Starting multi-season training demo")
    
    # Step 1: Create necessary directories
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('models/results', exist_ok=True)
    
    # Step 2: Collect data from multiple seasons (2021-2023)
    logger.info("Collecting data from the 2021, 2022, and 2023 MLB seasons")
    collection_cmd = [
        sys.executable, 
        'src/collect_seasons.py', 
        '--seasons', '2021', '2022', '2023',
        '--skip-existing'  # Skip re-downloading if data already exists
    ]
    
    try:
        subprocess.run(collection_cmd, check=True)
        logger.info("Multi-season data collection completed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Data collection failed: {str(e)}")
        return 1
    
    # Step 3: Run the pipeline with the collected data
    logger.info("Processing data and training models")
    pipeline_cmd = [
        sys.executable, 
        'run_pipeline.py',
        '--skip-collection'  # Skip collection since we already collected the data
    ]
    
    try:
        subprocess.run(pipeline_cmd, check=True)
        logger.info("Pipeline completed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Pipeline failed: {str(e)}")
        return 1
    
    logger.info("\n======= Multi-Season Training Complete =======")
    logger.info("The models have been trained using data from 2021-2023 MLB seasons")
    logger.info("You can find the trained models in the 'models/' directory")
    logger.info("Model evaluation results are in 'models/results/'")
    logger.info("\nTo make predictions with your trained model, run:")
    logger.info("python src/predict.py --interactive")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 