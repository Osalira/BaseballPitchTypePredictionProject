# Baseball Pitch Prediction Application

This application predicts the type of pitch a pitcher will throw next based on the game situation, pitcher tendencies, and batter performance. It uses machine learning to provide insights into what pitch is most likely coming next.

## Features

- **Real-time Pitch Prediction**: Predicts the type of pitch (fastball, slider, curveball, etc.) based on current game situation
- **Interactive Dashboard**: Visualizes prediction accuracy and trends
- **Pitcher and Batter Analysis**: Allows users to input specific pitcher and batter information
- **Prediction History**: Keeps track of past predictions for analysis
- **Actual Outcome Tracking**: Records the actual pitches thrown to continually improve the model

## Technical Stack

- **Frontend**: Vue.js with Tailwind CSS
- **Backend**: Python (Flask API)
- **Database**: SQLite for local data storage, Firebase Firestore for cloud storage
- **Machine Learning**: XGBoost model for pitch type prediction

## Actual Outcome Tracking

One of the key features of this application is its ability to learn from user feedback. Here's how it works:

1. The system makes a prediction about the next pitch type
2. After the pitch is thrown, users can record the actual pitch outcome
3. This feedback is stored in Firebase Firestore and synced back to SQLite
4. The model retraining system incorporates this feedback data to improve prediction accuracy

### Benefits of Feedback Loop

- Continuous model improvement based on real-world outcomes
- Adaptation to changing pitcher tendencies and strategies
- Increased prediction accuracy over time
- Personalized predictions based on user-specific data

## Model Retraining

The application includes an automated model retraining system that:

1. Collects user feedback on actual pitch outcomes
2. Integrates this feedback with the original training data
3. Retrains the model periodically (configurable schedule)
4. Deploys updated models automatically

This creates a virtuous cycle where user engagement improves the model, which leads to better predictions, which increases user engagement.

## Getting Started

### Prerequisites

- Node.js and npm for frontend
- Python 3.8+ for backend
- Firebase account for cloud database

### Installation

1. Clone the repository
2. Set up the backend:
   ```
   cd backend
   pip install -r requirements.txt
   ```

3. Set up the frontend:
   ```
   cd frontend/pitch-prediction-ui
   npm install
   ```

4. Configure Firebase:
   - Create a Firebase project
   - Set up Firestore Database
   - Add your Firebase configuration to `.env` files in both frontend and backend

### Running the Application

1. Start the backend server:
   ```
   python app.py
   ```

2. Start the frontend development server:
   ```
   cd frontend/pitch-prediction-ui
   npm run dev
   ```

3. Access the application at `http://localhost:5173`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

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

## Database Setup

For optimized performance, the project uses an SQLite database to store and process baseball data efficiently:

1. Initialize the database structure:
   ```
   python setup_database.py
   ```

2. This creates a `data/baseball.db` file with tables for:
   - Metadata tracking
   - Optimized indices for faster queries
   - Firebase synchronization tracking (for optional web frontend)

3. The database enables:
   - Faster data loading and processing
   - Reduced memory usage during preprocessing
   - Better handling of large multi-season datasets
   - Integration with web frontends through Firebase

## Multi-Season Training

This project supports training with data from multiple MLB seasons to improve model accuracy:

1. Collect data from multiple seasons:
   ```
   python src/collect_seasons.py --seasons 2021 2022 2023
   ```

2. Train models using the collected data:
   ```
   python run_pipeline.py --skip-collection
   ```

3. Or use the all-in-one demo script:
   ```
   python multi_season_demo.py
   ```

## Features

The model will consider the following features for prediction:
- Fastball percentage for each pitcher (overall and by count)
- Fastball percentage for each catcher
- Batter's performance against different pitch types
- Game situation (count, outs, runners, score)
- Inning and game context
- Season information (when using multi-season data)

## Model Development Plan

1. Data Collection: Gather MLB data using pybaseball
2. Data Preprocessing: Clean and prepare features
3. Exploratory Data Analysis: Understand patterns and relationships
4. Feature Engineering: Create meaningful inputs for the model
5. Model Training: Test various algorithms (logistic regression, decision trees, XGBoost, etc.)
6. Evaluation: Assess models using confusion matrices and accuracy metrics
7. Deployment: Prepare model for real-time prediction

## Making Predictions

To make predictions with a trained model:

```
python src/predict.py --interactive
```

This will open an interactive session where you can input game situations and get pitch predictions.

## Custom Data Collection

To collect data for a specific date range:

```
python run_pipeline.py --start-date "2023-04-01" --end-date "2023-10-01" 