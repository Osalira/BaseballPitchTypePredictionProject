# Getting Started with Baseball Pitch Prediction

This is a quick start guide for the Baseball Pitch Prediction project. For more detailed information, please refer to the [Project Guide](PROJECT_GUIDE.md).

## Quick Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Complete Pipeline

```bash
python run_pipeline.py
```

This will:
- Collect MLB pitch data from the 2022 season
- Preprocess the data and create features
- Train multiple machine learning models
- Save all models and evaluation results

## Data Sources

By default, the project collects MLB Statcast data using the `pybaseball` library. This includes:
- Pitch type information
- Game situations (count, outs, inning)
- Pitcher and batter information
- Pitch outcomes

## Making Predictions

### Interactive Mode

```bash
python src/predict.py --interactive
```

Follow the prompts to enter game situation information and get pitch predictions.

### Command Line Mode

```bash
python src/predict.py --balls 1 --strikes 2 --outs 1 --inning 5 --pitcher-fb-pct 0.6
```

## Exploring the Data and Results

Jupyter notebooks are provided for data exploration and model evaluation:

```bash
jupyter notebook
```

Open:
- `notebooks/01_exploratory_data_analysis.ipynb` - to explore the data
- `notebooks/02_model_evaluation.ipynb` - to evaluate model performance

## Customizing the Pipeline

You can customize the data collection period:

```bash
python run_pipeline.py --start-date "2023-04-01" --end-date "2023-10-01"
```

Or skip certain steps:

```bash
# Skip data collection (use existing data)
python run_pipeline.py --skip-collection

# Skip collection and preprocessing (just train models)
python run_pipeline.py --skip-collection --skip-preprocessing
```

## Recommended Workflow for Real Games

1. **Before the game**: 
   - Collect data on opposing pitchers
   - Train models on this data

2. **During the game**:
   - Run the prediction tool in interactive mode
   - Input current game situation
   - Use prediction to inform batting strategy

3. **After the game**:
   - Record actual pitches thrown
   - Update your dataset with new observations
   - Retrain models to improve future predictions 