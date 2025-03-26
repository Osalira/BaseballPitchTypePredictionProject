# Baseball Pitch Prediction

Machine learning model to predict whether the next pitch will be a fastball or an offspeed pitch in baseball.

## Project Overview

This project aims to develop a machine learning model capable of predicting the type of pitch (fastball or offspeed) that will be thrown in a given baseball game situation. The model will utilize various features including pitcher tendencies, catcher trends, game situation data, and historical pitch patterns.

## Project Structure

- `data/raw/`: Raw MLB data collected using pybaseball
- `data/processed/`: Cleaned and preprocessed data ready for modeling
- `notebooks/`: Jupyter notebooks for exploration, analysis, and visualization
- `src/`: Python source code for data processing and modeling
- `models/`: Saved trained models

## Setup

1. Install required packages:
   ```
   pip install -r requirements.txt
   ```

2. Run the data collection script:
   ```
   python src/data_collection.py
   ```

## Features

The model will consider the following features for prediction:
- Fastball percentage for each pitcher (overall and by count)
- Fastball percentage for each catcher
- Batter's performance against different pitch types
- Game situation (count, outs, runners, score)
- Inning and game context

## Model Development Plan

1. Data Collection: Gather MLB data using pybaseball
2. Data Preprocessing: Clean and prepare features
3. Exploratory Data Analysis: Understand patterns and relationships
4. Feature Engineering: Create meaningful inputs for the model
5. Model Training: Test various algorithms (logistic regression, decision trees, XGBoost, etc.)
6. Evaluation: Assess models using confusion matrices and accuracy metrics
7. Deployment: Prepare model for real-time prediction 