# Baseball Pitch Prediction - Project Guide

This guide provides comprehensive information on how to use this project for predicting whether the next pitch in a baseball game will be a fastball or an offspeed pitch.

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Data Collection](#data-collection)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Model Training](#model-training)
- [Making Predictions](#making-predictions)
- [Running the Complete Pipeline](#running-the-complete-pipeline)
- [Extending the Project](#extending-the-project)
- [FAQ](#faq)

## Project Overview

This project applies machine learning to predict the type of pitch (fastball or offspeed) that will be thrown in a given baseball game situation. The model uses various features including:

- Game situation (count, outs, inning)
- Pitcher tendencies (overall fastball percentage, fastball percentage by count)
- Batter performance (success against fastball vs. offspeed pitches)
- Previous pitch information (sequence patterns)

The goal is to provide a practical tool that baseball players and teams can use to gain insights into opponent strategies and improve their decision-making.

## Installation

### Prerequisites

- Python 3.8+ installed
- Git (optional)

### Step 1: Set up the environment

```bash
# Clone the repository (if using Git)
git clone https://github.com/yourusername/pitch_prediction.git
cd pitch_prediction

# Create and activate a virtual environment (recommended)
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python -m venv venv
source venv/bin/activate
```

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

## Project Structure

```
pitch_prediction/
├── data/                # Data directory
│   ├── raw/             # Raw MLB data collected with pybaseball
│   └── processed/       # Processed data ready for modeling
├── models/              # Trained models
│   └── results/         # Model evaluation results
├── notebooks/           # Jupyter notebooks
│   ├── 01_exploratory_data_analysis.ipynb
│   └── 02_model_evaluation.ipynb
├── src/                 # Source code
│   ├── data_collection.py
│   ├── data_preprocessing.py
│   ├── model_training.py
│   └── predict.py
├── run_pipeline.py      # Script to run the complete pipeline
├── requirements.txt     # Project dependencies
├── README.md            # Project overview
└── PROJECT_GUIDE.md     # This detailed guide
```

## Data Collection

The project uses the `pybaseball` library to collect MLB Statcast data. 

### Running Data Collection

```bash
python src/data_collection.py
```

This script will:
1. Collect Statcast pitch data for the specified date range (default: 2022 MLB season)
2. Save the raw data in `data/raw/` directory
3. Add a binary classification column to identify fastball vs offspeed pitches
4. Save the classified data in `data/processed/` directory

### Customizing Data Collection

To collect data for a different date range, modify the following variables in `src/data_collection.py`:

```python
start_date = '2022-04-07'  # Opening day 2022
end_date = '2022-10-05'    # Last day of 2022 regular season
```

Or use the pipeline script with date parameters:

```bash
python run_pipeline.py --start-date "2023-04-01" --end-date "2023-10-01"
```

## Data Preprocessing

After collecting the raw data, the preprocessing step creates features needed for prediction.

### Running Data Preprocessing

```bash
python src/data_preprocessing.py
```

This script will:
1. Clean the data by handling missing values and filtering out invalid records
2. Create game situation features (count type, inning stage, etc.)
3. Calculate pitcher tendencies (fastball percentage overall and by count)
4. Calculate batter tendencies (success against different pitch types)
5. Create sequence features based on previous pitches
6. Prepare the final modeling dataset

## Exploratory Data Analysis

The project includes Jupyter notebooks for exploring the data and understanding patterns:

### Running the EDA Notebook

```bash
# Start Jupyter notebook
jupyter notebook
```

Then open `notebooks/01_exploratory_data_analysis.ipynb`

This notebook will help you:
- Understand the distribution of fastball vs offspeed pitches
- Analyze how count affects pitch selection
- Explore pitcher tendencies in different situations
- Visualize sequence patterns

## Model Training

The project trains multiple machine learning models to predict pitch type.

### Running Model Training

```bash
python src/model_training.py
```

This script will:
1. Load the preprocessed modeling data
2. Split the data into training and test sets
3. Train multiple models:
   - Logistic Regression (baseline model)
   - Decision Tree (interpretable model)
   - Random Forest
   - XGBoost
4. Evaluate each model on the test set
5. Save the trained models and evaluation results

### Model Evaluation

To analyze model performance in detail, use the model evaluation notebook:

```bash
jupyter notebook notebooks/02_model_evaluation.ipynb
```

This notebook provides:
- Detailed comparison of model metrics
- Feature importance analysis
- ROC curve comparison
- SHAP analysis for model interpretation
- Sample predictions for common game scenarios

## Making Predictions

The project provides a script for making real-time predictions during a game.

### Running the Prediction Tool

#### Interactive Mode

```bash
python src/predict.py --interactive
```

This will start an interactive session where you can input game situations and get pitch predictions.

#### Command Line Mode

```bash
python src/predict.py --balls 1 --strikes 2 --outs 1 --inning 5 --pitcher-fb-pct 0.6 --batter-vs-fb 0.3 --batter-vs-os 0.25
```

This mode is useful for automated predictions or integration with other systems.

## Running the Complete Pipeline

The project includes a script to run the entire pipeline from data collection to prediction.

```bash
python run_pipeline.py
```

### Pipeline Options

```
--start-date DATE    Start date for data collection (YYYY-MM-DD)
--end-date DATE      End date for data collection (YYYY-MM-DD)
--skip-collection    Skip data collection step
--skip-preprocessing Skip data preprocessing step
--skip-training      Skip model training step
--interactive        Run interactive prediction at the end
```

Example usage:

```bash
# Run only preprocessing and training (skip collection)
python run_pipeline.py --skip-collection

# Run full pipeline with custom date range and end with interactive prediction
python run_pipeline.py --start-date "2023-04-01" --end-date "2023-10-01" --interactive
```

## Extending the Project

### Adding New Features

To add new features to the model:

1. Modify the `src/data_preprocessing.py` script to create your new features
2. Update the `model_features` list in the `prepare_modeling_data` function to include your new features
3. Retrain the models

### Training on Different Leagues

To adapt the project for other leagues:

1. Modify the data collection script to collect data from your target league, or manually prepare a dataset with the same structure
2. Ensure the pitch types are properly classified into fastball/offspeed categories
3. Run the preprocessing and training steps with your new data

### Predicting Specific Pitch Types

To extend the model to predict specific pitch types (not just fastball/offspeed):

1. Modify the `categorize_pitches` function in `src/data_collection.py` to create a multi-class label
2. Update the models in `src/model_training.py` to handle multi-class classification
3. Modify the evaluation metrics and reporting for multi-class problems

## FAQ

### How accurate is the model?

The model typically achieves 65-75% accuracy on test data, depending on the algorithm used. XGBoost generally performs the best, while the Decision Tree provides more interpretable results.

### What features are most important for prediction?

The most important features typically include:
- Count (balls and strikes)
- Pitcher's overall fastball percentage
- Previous pitch type
- Inning and game situation

### How can I use this for my baseball team?

1. Collect data on opposing pitchers during games
2. Run the interactive prediction tool during your games
3. Use the insights to inform your batting approach

### Does this work for all levels of baseball?

The model is trained on MLB data, but the principles apply to all levels of baseball. For best results at other levels:
1. Collect data specific to your league
2. Retrain the model on that data
3. Validate the predictions during actual games

### How do I handle missing or incomplete data?

The preprocessing script handles missing values by:
- Dropping rows with missing essential data (pitch type, count, etc.)
- Using median values for missing numeric features
- Creating special flags for unknown or missing categorical values

For real-time prediction with incomplete data, the system uses reasonable defaults where possible.

---

For additional questions or support, please open an issue on the project repository or contact the project maintainers. 