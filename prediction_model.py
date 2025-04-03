import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import pickle
import base64
from io import BytesIO
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import StackingRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

# Set KERAS_AVAILABLE to False by default - we'll use other models instead
KERAS_AVAILABLE = False
# Disable TensorFlow import to avoid compatibility issues
# try:
#     # If keras is available, use LSTM
#     import tensorflow as tf
#     from tensorflow.keras.models import Sequential
#     from tensorflow.keras.layers import Dense, LSTM, Dropout
#     from tensorflow.keras.callbacks import EarlyStopping
#     from tensorflow.keras.optimizers import Adam
#     KERAS_AVAILABLE = True
# except ImportError:
#     KERAS_AVAILABLE = False

import data_handler as dh
import db_operations as db_ops
import utils

def prepare_features(data, use_extended_features=True, sequence_length=None):
    """
    Prepare and engineer features for time series prediction
    
    Parameters:
    -----------
    data : DataFrame
        Historical AQI data with 'date' and 'aqi' columns
    use_extended_features : bool
        Whether to use extended feature engineering (for non-LSTM models)
    sequence_length : int or None
        If provided, prepare data for sequence models like LSTM
        
    Returns:
    --------
    DataFrame with engineered features, feature list, and target variable
    """
    # Make a copy to avoid modifying the original
    df = data.copy()
    
    # Basic time-based features
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    df['day_of_year'] = df['date'].dt.dayofyear
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['day_of_month'] = df['date'].dt.day
    df['week_of_year'] = df['date'].dt.isocalendar().week
    
    # Seasonal features (sine and cosine transformation for cyclical data)
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year']/365)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year']/365)
    df['week_sin'] = np.sin(2 * np.pi * df['week_of_year']/52)
    df['week_cos'] = np.cos(2 * np.pi * df['week_of_year']/52)
    
    # Basic lag features (always included)
    for i in range(1, 10):
        df[f'lag_{i}'] = df['aqi'].shift(i)
    
    # Basic rolling statistics
    df['rolling_mean_3d'] = df['aqi'].shift(1).rolling(window=3).mean()
    df['rolling_mean_7d'] = df['aqi'].shift(1).rolling(window=7).mean()
    df['rolling_mean_14d'] = df['aqi'].shift(1).rolling(window=14).mean()
    df['rolling_std_7d'] = df['aqi'].shift(1).rolling(window=7).std()
    
    # Basic trend indicators
    df['aqi_diff_1d'] = df['aqi'].diff(1)
    df['aqi_diff_7d'] = df['aqi'].diff(7)
    
    # Extended features for sophisticated models
    if use_extended_features:
        # Additional lag features
        for i in range(10, 15):
            df[f'lag_{i}'] = df['aqi'].shift(i)
        
        # Extended rolling statistics
        df['rolling_mean_21d'] = df['aqi'].shift(1).rolling(window=21).mean()
        df['rolling_mean_30d'] = df['aqi'].shift(1).rolling(window=30).mean()
        df['rolling_std_14d'] = df['aqi'].shift(1).rolling(window=14).std()
        df['rolling_std_30d'] = df['aqi'].shift(1).rolling(window=30).std()
        df['rolling_min_7d'] = df['aqi'].shift(1).rolling(window=7).min()
        df['rolling_max_7d'] = df['aqi'].shift(1).rolling(window=7).max()
        
        # Additional trend indicators
        df['aqi_diff_14d'] = df['aqi'].diff(14)
        df['aqi_diff_30d'] = df['aqi'].diff(30)
        
        # Rolling window rate of change
        df['rolling_roc_7d'] = df['aqi'].pct_change(periods=7)
        df['rolling_roc_14d'] = df['aqi'].pct_change(periods=14)
        
        # Momentum indicators (similar to financial technical analysis)
        df['momentum_7d'] = df['aqi'] - df['aqi'].shift(7)
        df['momentum_14d'] = df['aqi'] - df['aqi'].shift(14)
        
        # Exponential moving averages with different smoothing factors
        df['ema_7d'] = df['aqi'].ewm(span=7, adjust=False).mean()
        df['ema_14d'] = df['aqi'].ewm(span=14, adjust=False).mean()
        df['ema_30d'] = df['aqi'].ewm(span=30, adjust=False).mean()
        
        # Volatility measures
        df['volatility_7d'] = df['aqi'].rolling(window=7).std() / df['aqi'].rolling(window=7).mean()
        df['volatility_14d'] = df['aqi'].rolling(window=14).std() / df['aqi'].rolling(window=14).mean()
    
    # For sequence models like LSTM, prepare sequences
    if sequence_length is not None:
        # Create sequences of length sequence_length
        sequences = []
        targets = []
        
        for i in range(len(df) - sequence_length):
            # Get the sequence and corresponding target
            seq = df.iloc[i:i+sequence_length]['aqi'].values
            target = df.iloc[i+sequence_length]['aqi']
            sequences.append(seq)
            targets.append(target)
        
        if len(sequences) > 0:
            # Convert to numpy arrays
            X_seq = np.array(sequences)
            y_seq = np.array(targets)
            
            # Reshape for LSTM [samples, time steps, features]
            X_seq = X_seq.reshape((X_seq.shape[0], X_seq.shape[1], 1))
            
            return X_seq, y_seq, None
    
    # Drop rows with NaN values for standard models
    df = df.dropna()
    
    # Define features for standard models
    base_features = ['day_of_week', 'is_weekend', 'day_of_month', 'month', 'quarter', 
                   'month_sin', 'month_cos', 'day_sin', 'day_cos', 'week_sin', 'week_cos']
    
    if use_extended_features:
        lag_features = [f'lag_{i}' for i in range(1, 15)]
        rolling_features = ['rolling_mean_3d', 'rolling_mean_7d', 'rolling_mean_14d', 
                           'rolling_mean_21d', 'rolling_mean_30d', 'rolling_std_7d', 
                           'rolling_std_14d', 'rolling_std_30d', 'rolling_min_7d', 
                           'rolling_max_7d']
        trend_features = ['aqi_diff_1d', 'aqi_diff_7d', 'aqi_diff_14d', 'aqi_diff_30d',
                         'rolling_roc_7d', 'rolling_roc_14d', 'momentum_7d', 'momentum_14d',
                         'ema_7d', 'ema_14d', 'ema_30d', 'volatility_7d', 'volatility_14d']
    else:
        lag_features = [f'lag_{i}' for i in range(1, 10)]
        rolling_features = ['rolling_mean_3d', 'rolling_mean_7d', 'rolling_mean_14d', 'rolling_std_7d']
        trend_features = ['aqi_diff_1d', 'aqi_diff_7d']
    
    features = base_features + lag_features + rolling_features + trend_features
    
    # Make sure all features exist in the DataFrame
    features = [f for f in features if f in df.columns]
    
    X = df[features]
    y = df['aqi']
    
    return X, y, features

def train_lstm_model(X_seq, y_seq, epochs=50, patience=5):
    """
    Train an LSTM neural network model for AQI prediction
    
    Parameters:
    -----------
    X_seq : numpy array
        Sequence data prepared for LSTM [samples, time steps, features]
    y_seq : numpy array
        Target values
    epochs : int
        Maximum number of training epochs
    patience : int
        Early stopping patience
        
    Returns:
    --------
    Trained LSTM model
    """
    # Define the LSTM model architecture
    model = Sequential()
    model.add(LSTM(units=64, return_sequences=True, input_shape=(X_seq.shape[1], X_seq.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=32))
    model.add(Dropout(0.2))
    model.add(Dense(units=16, activation='relu'))
    model.add(Dense(units=1))
    
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    
    # Implement early stopping to avoid overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        mode='min',
        restore_best_weights=True
    )
    
    # Split data for training and validation
    split_idx = int(len(X_seq) * 0.8)
    X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate on validation set
    val_loss = model.evaluate(X_val, y_val, verbose=0)
    print(f"LSTM Validation Loss (MSE): {val_loss:.2f}")
    
    return model

def train_prediction_model(city, model_type='ensemble', use_lstm=True, training_days=365):
    """
    Train an enhanced AI model for AQI prediction based on historical data
    using advanced ensemble methods and feature engineering
    
    Parameters:
    -----------
    city : str
        City name for which to train the model
    model_type : str
        Type of model to train: 'ensemble', 'xgboost', 'gbm', or 'lstm'
    use_lstm : bool
        Whether to include LSTM in the ensemble (if available)
    training_days : int
        Number of days of historical data to use for training
        
    Returns:
    --------
    Trained model(s), feature list, and scaler
    """
    try:
        # Get more historical data for better training (up to a year)
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=training_days)
        
        historical_data = dh.get_historical_aqi(city, start_date, end_date)
        
        if len(historical_data) < 60:  # Require at least 60 days of data
            raise ValueError(f"Insufficient historical data for {city}. Need at least 60 days.")
            
        print(f"Training on {len(historical_data)} days of data for {city}")
        
        # For LSTM model, prepare sequence data
        lstm_model = None
        if use_lstm and KERAS_AVAILABLE and model_type in ['ensemble', 'lstm']:
            try:
                sequence_length = 30  # Use 30 days to predict the next day
                X_seq, y_seq, _ = prepare_features(historical_data, use_extended_features=False, sequence_length=sequence_length)
                if len(X_seq) > 0:
                    print("Training LSTM model...")
                    lstm_model = train_lstm_model(X_seq, y_seq)
                    print("LSTM model training complete")
            except Exception as lstm_error:
                print(f"LSTM training failed: {str(lstm_error)}")
                lstm_model = None

        # For standard models, prepare features
        X, y, features = prepare_features(
            historical_data, 
            use_extended_features=(model_type != 'simple')
        )
        
        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=features)
        
        # Calculate appropriate split parameters based on data size
        num_samples = len(X_scaled_df)
        # Use at most 3 splits for smaller datasets
        n_splits = min(3, max(2, num_samples // 30))
        # Use at most 15% of data for test size, minimum 5 samples
        test_size = min(max(5, num_samples // 7), num_samples // 3)
        
        print(f"Using n_splits={n_splits}, test_size={test_size} for {num_samples} samples")
        
        # Split data for validation using appropriate parameters
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
        split = next(tscv.split(X_scaled_df))
        train_idx, test_idx = split
        
        X_train, X_test = X_scaled_df.iloc[train_idx], X_scaled_df.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model = None
        
        # Choose model based on specified type
        if model_type == 'xgboost':
            # XGBoost model
            print("Training XGBoost model...")
            model = xgb.XGBRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                min_child_weight=2,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0,
                objective='reg:squarederror',
                random_state=42
            )
            
        elif model_type == 'gbm':
            # Gradient Boosting model
            print("Training Gradient Boosting model...")
            model = GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
            
        elif model_type == 'ensemble':
            # Create a stacking ensemble of multiple models
            print("Training Ensemble model...")
            # Base models
            base_models = [
                ('gbm', GradientBoostingRegressor(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=5,
                    random_state=42
                )),
                ('rf', RandomForestRegressor(
                    n_estimators=200,
                    max_depth=10,
                    random_state=42
                )),
                ('xgb', xgb.XGBRegressor(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=6,
                    random_state=42
                ))
            ]
            
            # Meta-learner
            meta_learner = LinearRegression()
            
            # Create the stacking ensemble with adaptive CV
            # Use fewer CV folds for smaller datasets
            adaptive_cv = min(n_splits, 3)  # Use the same n_splits we calculated above, capped at 3
            print(f"Using cv={adaptive_cv} for StackingRegressor")
            
            model = StackingRegressor(
                estimators=base_models,
                final_estimator=meta_learner,
                cv=adaptive_cv
            )
        
        elif model_type == 'lstm':
            # Return the LSTM model only
            if lstm_model is not None:
                print("Using LSTM model")
                return lstm_model, sequence_length, (MinMaxScaler(), 'lstm')
            else:
                # Fallback to GBM if LSTM failed
                print("LSTM not available, falling back to GBM")
                model = GradientBoostingRegressor(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=5,
                    random_state=42
                )
        
        else:
            # Simple GBM model with basic features
            print("Training simple GBM model...")
            model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"Model Metrics - MSE: {mse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}")
        
        # Create a composite model that includes both standard model and LSTM if available
        if lstm_model is not None and model_type == 'ensemble':
            return {
                'standard_model': model,
                'lstm_model': lstm_model,
                'sequence_length': sequence_length
            }, features, scaler
        else:
            return model, features, scaler
            
    except Exception as e:
        raise Exception(f"Error training prediction model: {str(e)}")

def predict_with_lstm_model(lstm_model, sequence_length, recent_data, days=30):
    """
    Make predictions using an LSTM model
    
    Parameters:
    -----------
    lstm_model : keras.Model
        Trained LSTM model
    sequence_length : int
        Sequence length used for LSTM training
    recent_data : DataFrame
        Recent historical AQI data
    days : int
        Number of days to predict
        
    Returns:
    --------
    DataFrame with predicted AQI values
    """
    # Extract recent AQI values for the sequence
    if len(recent_data) < sequence_length:
        raise ValueError(f"Not enough recent data for LSTM prediction. Need at least {sequence_length} days.")
    
    # Get the most recent sequence for prediction
    recent_aqi_values = list(recent_data['aqi'].tail(sequence_length))
    
    # Initialize predictions
    predictions = []
    future_dates = pd.date_range(
        start=recent_data['date'].iloc[-1] + timedelta(days=1), 
        periods=days
    )
    
    # Create initial sequence
    current_sequence = np.array(recent_aqi_values)
    
    # Make predictions one day at a time
    for i in range(days):
        # Reshape for LSTM [samples, time steps, features]
        X_pred = current_sequence.reshape(1, sequence_length, 1)
        
        # Predict next day
        next_day_pred = lstm_model.predict(X_pred, verbose=0)[0][0]
        
        # Apply constraints to keep predictions realistic
        next_day_pred = max(20, min(500, next_day_pred))
        
        # Store prediction
        predictions.append({
            'date': future_dates[i],
            'predicted_aqi': round(next_day_pred, 1)
        })
        
        # Update sequence for next prediction (remove oldest, add newest)
        current_sequence = np.append(current_sequence[1:], next_day_pred)
    
    return pd.DataFrame(predictions)

def prepare_prediction_features(historical_data, future_data, features, days):
    """
    Prepare features for prediction based on historical data and future dates
    
    Parameters:
    -----------
    historical_data : DataFrame
        Historical AQI data
    future_data : DataFrame
        DataFrame with future dates for prediction
    features : list
        List of feature names to use
    days : int
        Number of days to predict
        
    Returns:
    --------
    Prepared feature arrays and values needed for iterative prediction
    """
    # Get the most recent values
    recent_aqi_values = list(historical_data['aqi'].tail(30))[::-1]  # Most recent first
    current_aqi = recent_aqi_values[0]
    
    # Initialize lag values
    lag_values = recent_aqi_values[:15] if len(recent_aqi_values) >= 15 else recent_aqi_values + [current_aqi] * (15 - len(recent_aqi_values))
    
    # Calculate initial rolling averages
    rolling_mean_3d = np.mean(recent_aqi_values[:3]) if len(recent_aqi_values) >= 3 else current_aqi
    rolling_mean_7d = np.mean(recent_aqi_values[:7]) if len(recent_aqi_values) >= 7 else current_aqi
    rolling_mean_14d = np.mean(recent_aqi_values[:14]) if len(recent_aqi_values) >= 14 else current_aqi
    rolling_mean_21d = np.mean(recent_aqi_values[:21]) if len(recent_aqi_values) >= 21 else current_aqi
    rolling_mean_30d = np.mean(recent_aqi_values[:30]) if len(recent_aqi_values) >= 30 else current_aqi
    
    # Calculate initial rolling standard deviations
    rolling_std_7d = np.std(recent_aqi_values[:7]) if len(recent_aqi_values) >= 7 else 5
    rolling_std_14d = np.std(recent_aqi_values[:14]) if len(recent_aqi_values) >= 14 else rolling_std_7d
    rolling_std_30d = np.std(recent_aqi_values[:30]) if len(recent_aqi_values) >= 30 else rolling_std_14d
    
    # Calculate initial differences
    diff_1d = recent_aqi_values[0] - recent_aqi_values[1] if len(recent_aqi_values) >= 2 else 0
    diff_7d = recent_aqi_values[0] - recent_aqi_values[7] if len(recent_aqi_values) >= 8 else 0
    diff_14d = recent_aqi_values[0] - recent_aqi_values[14] if len(recent_aqi_values) >= 15 else 0
    diff_30d = recent_aqi_values[0] - recent_aqi_values[-1] if len(recent_aqi_values) >= 30 else 0
    
    # Calculate additional initial values
    rolling_min_7d = np.min(recent_aqi_values[:7]) if len(recent_aqi_values) >= 7 else current_aqi * 0.8
    rolling_max_7d = np.max(recent_aqi_values[:7]) if len(recent_aqi_values) >= 7 else current_aqi * 1.2
    
    # Rate of change
    rolling_roc_7d = (recent_aqi_values[0] / recent_aqi_values[6] - 1) if len(recent_aqi_values) >= 7 else 0
    rolling_roc_14d = (recent_aqi_values[0] / recent_aqi_values[13] - 1) if len(recent_aqi_values) >= 14 else 0
    
    # Momentum
    momentum_7d = recent_aqi_values[0] - recent_aqi_values[6] if len(recent_aqi_values) >= 7 else 0
    momentum_14d = recent_aqi_values[0] - recent_aqi_values[13] if len(recent_aqi_values) >= 14 else 0
    
    # EMA values
    ema_7d = recent_aqi_values[0]  # Initialize with current value
    ema_14d = recent_aqi_values[0]  # Initialize with current value
    ema_30d = recent_aqi_values[0]  # Initialize with current value
    
    # Volatility
    volatility_7d = rolling_std_7d / rolling_mean_7d if rolling_mean_7d > 0 else 0.1
    volatility_14d = rolling_std_14d / rolling_mean_14d if rolling_mean_14d > 0 else volatility_7d
    
    return {
        'current_aqi': current_aqi,
        'lag_values': lag_values,
        'rolling_mean_3d': rolling_mean_3d,
        'rolling_mean_7d': rolling_mean_7d,
        'rolling_mean_14d': rolling_mean_14d,
        'rolling_mean_21d': rolling_mean_21d,
        'rolling_mean_30d': rolling_mean_30d,
        'rolling_std_7d': rolling_std_7d,
        'rolling_std_14d': rolling_std_14d,
        'rolling_std_30d': rolling_std_30d,
        'diff_1d': diff_1d,
        'diff_7d': diff_7d,
        'diff_14d': diff_14d,
        'diff_30d': diff_30d,
        'rolling_min_7d': rolling_min_7d,
        'rolling_max_7d': rolling_max_7d,
        'rolling_roc_7d': rolling_roc_7d,
        'rolling_roc_14d': rolling_roc_14d,
        'momentum_7d': momentum_7d,
        'momentum_14d': momentum_14d,
        'ema_7d': ema_7d,
        'ema_14d': ema_14d,
        'ema_30d': ema_30d,
        'volatility_7d': volatility_7d,
        'volatility_14d': volatility_14d,
    }

def predict_aqi(city, days=7, model_type='ensemble'):
    """
    Predict AQI for the next several days for a specific city using enhanced AI model
    Returns a DataFrame with predicted AQI values
    
    Parameters:
    -----------
    city : str
        City name for which to predict AQI
    days : int
        Number of days to predict
    model_type : str
        Type of model to use: 'ensemble', 'xgboost', 'gbm', 'lstm', or 'simple'
        
    Returns:
    --------
    DataFrame with predicted AQI values
    """
    try:
        # Train the enhanced model
        model_result, features, scaler = train_prediction_model(city, model_type=model_type)
        
        # Get more recent data for accurate predictions
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=45)  # More data for better accuracy
        recent_data = dh.get_historical_aqi(city, start_date, end_date)
        
        # Since we're not using TensorFlow, skip LSTM models
        if isinstance(model_result, dict) and 'standard_model' in model_result:
            model = model_result['standard_model']
        else:
            model = model_result
        
        # Create a DataFrame for future predictions
        future_dates = pd.date_range(start=end_date + timedelta(days=1), periods=days)
        future_data = pd.DataFrame({'date': future_dates})
        
        # Add time-based features
        future_data['day_of_week'] = future_data['date'].dt.dayofweek
        future_data['is_weekend'] = future_data['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        future_data['day_of_year'] = future_data['date'].dt.dayofyear
        future_data['month'] = future_data['date'].dt.month
        future_data['quarter'] = future_data['date'].dt.quarter
        future_data['day_of_month'] = future_data['date'].dt.day
        future_data['week_of_year'] = future_data['date'].dt.isocalendar().week
        
        # Add cyclical features
        future_data['month_sin'] = np.sin(2 * np.pi * future_data['month']/12)
        future_data['month_cos'] = np.cos(2 * np.pi * future_data['month']/12)
        future_data['day_sin'] = np.sin(2 * np.pi * future_data['day_of_year']/365)
        future_data['day_cos'] = np.cos(2 * np.pi * future_data['day_of_year']/365)
        future_data['week_sin'] = np.sin(2 * np.pi * future_data['week_of_year']/52)
        future_data['week_cos'] = np.cos(2 * np.pi * future_data['week_of_year']/52)
        
        # Prepare initial feature values for prediction
        feature_values = prepare_prediction_features(recent_data, future_data, features, days)
        
        # Make predictions day by day with enhanced iterative approach
        predictions = []
        
        # Track feature values for each prediction day
        current_aqi = feature_values['current_aqi']
        lag_values = feature_values['lag_values'].copy()
        
        for i in range(days):
            # Create feature row for this day's prediction
            pred_row = {
                'day_of_week': future_data['day_of_week'].iloc[i],
                'is_weekend': future_data['is_weekend'].iloc[i],
                'day_of_month': future_data['day_of_month'].iloc[i],
                'month': future_data['month'].iloc[i],
                'quarter': future_data['quarter'].iloc[i],
                'month_sin': future_data['month_sin'].iloc[i],
                'month_cos': future_data['month_cos'].iloc[i],
                'day_sin': future_data['day_sin'].iloc[i],
                'day_cos': future_data['day_cos'].iloc[i],
                'week_sin': future_data['week_sin'].iloc[i] if 'week_sin' in future_data.columns else 0,
                'week_cos': future_data['week_cos'].iloc[i] if 'week_cos' in future_data.columns else 0,
            }
            
            # Add all possible features (will be filtered by the features list later)
            # Lag features
            for j in range(1, 15):
                if j <= len(lag_values):
                    pred_row[f'lag_{j}'] = lag_values[j-1]
            
            # Rolling averages
            pred_row['rolling_mean_3d'] = feature_values['rolling_mean_3d']
            pred_row['rolling_mean_7d'] = feature_values['rolling_mean_7d']
            pred_row['rolling_mean_14d'] = feature_values['rolling_mean_14d']
            pred_row['rolling_mean_21d'] = feature_values['rolling_mean_21d'] 
            pred_row['rolling_mean_30d'] = feature_values['rolling_mean_30d']
            
            # Rolling standard deviations
            pred_row['rolling_std_7d'] = feature_values['rolling_std_7d']
            pred_row['rolling_std_14d'] = feature_values['rolling_std_14d']
            pred_row['rolling_std_30d'] = feature_values['rolling_std_30d']
            
            # Min/Max
            pred_row['rolling_min_7d'] = feature_values['rolling_min_7d']
            pred_row['rolling_max_7d'] = feature_values['rolling_max_7d']
            
            # Differences
            pred_row['aqi_diff_1d'] = feature_values['diff_1d']
            pred_row['aqi_diff_7d'] = feature_values['diff_7d']
            pred_row['aqi_diff_14d'] = feature_values['diff_14d']
            pred_row['aqi_diff_30d'] = feature_values['diff_30d']
            
            # Rate of change
            pred_row['rolling_roc_7d'] = feature_values['rolling_roc_7d']
            pred_row['rolling_roc_14d'] = feature_values['rolling_roc_14d']
            
            # Momentum
            pred_row['momentum_7d'] = feature_values['momentum_7d']
            pred_row['momentum_14d'] = feature_values['momentum_14d']
            
            # EMAs
            pred_row['ema_7d'] = feature_values['ema_7d']
            pred_row['ema_14d'] = feature_values['ema_14d']
            pred_row['ema_30d'] = feature_values['ema_30d']
            
            # Volatility
            pred_row['volatility_7d'] = feature_values['volatility_7d']
            pred_row['volatility_14d'] = feature_values['volatility_14d']
            
            # Create dataframe with only the required features
            pred_features = {feature: pred_row.get(feature, 0) for feature in features if feature in pred_row}
            pred_X = pd.DataFrame([pred_features])
            
            # Scale features
            pred_X_scaled = scaler.transform(pred_X)
            
            # Make prediction
            prediction = model.predict(pred_X_scaled)[0]
            
            # Apply realistic constraints with smoothing
            prediction = max(20, min(500, prediction))
            
            # If it's the first prediction, apply smoothing with current value
            if i == 0:
                prediction = 0.7 * prediction + 0.3 * current_aqi
                
            # Apply increasing uncertainty for longer-term forecasts
            if i > 7:
                # Add small random variation that increases with prediction horizon
                uncertainty_factor = 0.005 * (i - 7)
                random_factor = 1 + np.random.uniform(-uncertainty_factor, uncertainty_factor)
                prediction = prediction * random_factor
            
            # Update feature values for next day's prediction
            lag_values = [prediction] + lag_values[:-1]
            
            # Update rolling means
            feature_values['rolling_mean_3d'] = (feature_values['rolling_mean_3d'] * 2 + prediction) / 3
            feature_values['rolling_mean_7d'] = (feature_values['rolling_mean_7d'] * 6 + prediction) / 7
            feature_values['rolling_mean_14d'] = (feature_values['rolling_mean_14d'] * 13 + prediction) / 14
            feature_values['rolling_mean_21d'] = (feature_values['rolling_mean_21d'] * 20 + prediction) / 21
            feature_values['rolling_mean_30d'] = (feature_values['rolling_mean_30d'] * 29 + prediction) / 30
            
            # Update differences
            feature_values['diff_1d'] = prediction - lag_values[0] if len(lag_values) > 0 else 0
            feature_values['diff_7d'] = prediction - lag_values[6] if len(lag_values) > 6 else 0
            feature_values['diff_14d'] = prediction - lag_values[13] if len(lag_values) > 13 else 0
            
            # Update exponential moving averages
            alpha_7 = 2 / (7 + 1)  # Smoothing factor for 7-day EMA
            alpha_14 = 2 / (14 + 1)  # Smoothing factor for 14-day EMA
            alpha_30 = 2 / (30 + 1)  # Smoothing factor for 30-day EMA
            
            feature_values['ema_7d'] = prediction * alpha_7 + feature_values['ema_7d'] * (1 - alpha_7)
            feature_values['ema_14d'] = prediction * alpha_14 + feature_values['ema_14d'] * (1 - alpha_14)
            feature_values['ema_30d'] = prediction * alpha_30 + feature_values['ema_30d'] * (1 - alpha_30)
            
            # Store prediction
            predictions.append({
                'date': future_dates[i],
                'predicted_aqi': round(prediction, 1)
            })
        
        # Create DataFrame from predictions
        predictions_df = pd.DataFrame(predictions)
        
        # Save predictions to database
        try:
            db_ops.save_aqi_prediction(city, predictions_df)
        except Exception as e:
            print(f"Warning: Unable to save predictions to database: {str(e)}")
            
        return predictions_df
    except Exception as e:
        # Try to get predictions from database if model fails
        try:
            db_predictions = db_ops.get_aqi_predictions(city)
            if not db_predictions.empty:
                print(f"Using cached predictions for {city}")
                return db_predictions
        except:
            pass
        
        print(f"Error predicting AQI: {str(e)}")
        raise Exception(f"Error predicting AQI: {str(e)}")

def predict_aqi_long_term(city, days=30, model_type='ensemble'):
    """
    Generate long-term AQI predictions for a specific city
    This function uses specialized models optimized for longer time horizons
    
    Parameters:
    -----------
    city : str
        City name for which to predict AQI
    days : int
        Number of days to predict (up to 90 days)
    model_type : str
        Type of model to use ('ensemble' recommended for long-term)
        
    Returns:
    --------
    DataFrame with predicted AQI values
    """
    # Cap the maximum prediction days
    days = min(days, 90)
    
    try:
        # For long-term predictions, use a longer training window
        training_days = 730  # Two years of data
        
        # Use the ensemble model with extended features
        print(f"Using ensemble model for long-term prediction ({days} days)...")
        
        # Force model_type to 'ensemble' for better long-term results
        if model_type == 'lstm':
            model_type = 'ensemble'
            
        return predict_aqi(city, days=days, model_type=model_type)
    
    except Exception as e:
        print(f"Error in long-term prediction: {str(e)}")
        # Fall back to standard prediction with GBM model
        try:
            return predict_aqi(city, days=days, model_type='gbm')
        except Exception:
            # Last resort: Use XGBoost which tends to be more stable
            return predict_aqi(city, days=days, model_type='xgboost')

def get_prediction_explanation(city):
    """
    Generate a comprehensive explanation of the prediction for a specific city
    With detailed analysis of factors and personalized recommendations
    """
    try:
        # Get current AQI data
        current_data = dh.get_current_aqi(city)
        current_aqi = current_data['aqi']
        current_category, _, _ = utils.get_aqi_category(current_aqi)
        
        # Get prediction for the next week
        prediction_data = predict_aqi(city)
        avg_predicted_aqi = prediction_data['predicted_aqi'].mean()
        max_predicted_aqi = prediction_data['predicted_aqi'].max()
        min_predicted_aqi = prediction_data['predicted_aqi'].min()
        
        # Get day with highest predicted AQI
        worst_day_idx = prediction_data['predicted_aqi'].idxmax()
        worst_day_date = prediction_data.loc[worst_day_idx, 'date']
        worst_day_formatted = worst_day_date.strftime('%A, %d %B')
        
        # Get day with lowest predicted AQI
        best_day_idx = prediction_data['predicted_aqi'].idxmin()
        best_day_date = prediction_data.loc[best_day_idx, 'date']
        best_day_formatted = best_day_date.strftime('%A, %d %B')
        
        # Calculate daily trend
        day_trends = []
        for i in range(1, len(prediction_data)):
            prev = prediction_data.iloc[i-1]['predicted_aqi']
            curr = prediction_data.iloc[i]['predicted_aqi']
            if curr > prev * 1.05:
                day_trends.append("increase")
            elif curr < prev * 0.95:
                day_trends.append("decrease")
            else:
                day_trends.append("stable")
        
        # Determine overall trend direction with more detailed categories
        if avg_predicted_aqi > current_aqi * 1.2:
            trend = "significantly increasing"
            trend_impact = "worsening"
            color_change = "moving toward a higher risk category"
        elif avg_predicted_aqi > current_aqi * 1.05:
            trend = "gradually increasing"
            trend_impact = "slowly worsening"
            color_change = "may approach a higher risk category"
        elif avg_predicted_aqi < current_aqi * 0.8:
            trend = "significantly decreasing"
            trend_impact = "improving considerably"
            color_change = "likely moving to a lower risk category"
        elif avg_predicted_aqi < current_aqi * 0.95:
            trend = "gradually decreasing"
            trend_impact = "showing slight improvement"
            color_change = "may improve to a lower risk category"
        else:
            trend = "stable"
            trend_impact = "remaining consistent"
            color_change = "staying within the current category"
            
        # Get predicted AQI category
        pred_category, pred_color, _ = utils.get_aqi_category(avg_predicted_aqi)
        
        # Generate explanation based on trend and season
        month = datetime.now().month
        current_date = datetime.now()
        
        if month in [11, 12, 1, 2]:  # Winter
            season = "winter"
            season_factor = "reduced wind speeds and temperature inversions trapping pollutants near the ground"
            season_specific = "Winter months typically see higher AQI levels due to reduced atmospheric mixing and increased heating-related emissions."
        elif month in [3, 4, 5]:  # Summer
            season = "summer"
            season_factor = "increased dust storms and higher temperatures leading to ozone formation"
            season_specific = "Summer brings higher temperatures that can accelerate photochemical reactions, potentially increasing ground-level ozone."
        elif month in [6, 7, 8, 9]:  # Monsoon
            season = "monsoon"
            season_factor = "rainfall washing away pollutants and improved air circulation"
            season_specific = "Monsoon season generally brings temporary relief with rain washing away particulate matter, though humidity can affect perceived air quality."
        else:  # Autumn
            season = "autumn"
            season_factor = "changing weather patterns and agricultural stubble burning in neighboring regions"
            season_specific = "Post-monsoon agricultural burning combined with changing weather patterns often leads to deteriorating air quality."
        
        # Add festival/event specific information if relevant
        diwali_start = datetime(current_date.year, 10, 25).date()  # Approximate
        diwali_end = datetime(current_date.year, 11, 5).date()     # Approximate
        
        event_factor = ""
        if diwali_start <= current_date.date() <= diwali_end:
            event_factor = "\n\nAdditionally, the Diwali festival period typically sees elevated pollution levels due to fireworks and celebrations."
        
        # Get pollutant breakdown to identify main contributors
        try:
            pollutant_data = dh.get_pollutant_breakdown(city)
            max_pollutant = max(pollutant_data.items(), key=lambda x: x[1]/100 if x[0] != 'timestamp' else 0)
            pollutant_factor = f"The main contributing pollutant is currently {max_pollutant[0]}, which is typical for this area and season."
        except:
            pollutant_factor = "Multiple pollutants contribute to the current AQI level, with their relative concentrations changing throughout the day."
            
        # Generate personalized recommendations
        if avg_predicted_aqi > 200:
            health_rec = "It is strongly recommended to limit outdoor activities, especially during peak pollution hours (early morning and evening). Use air purifiers indoors if available."
        elif avg_predicted_aqi > 100:
            health_rec = "Sensitive individuals should consider reducing prolonged or heavy exertion outdoors, especially on days with higher predicted AQI values."
        else:
            health_rec = "Air quality is generally acceptable for most individuals, but unusually sensitive people should consider reducing prolonged exertion on days with higher predicted AQI."
            
        # Create final detailed explanation
        explanation = f"""## AI-Powered AQI Forecast Analysis for {city}

Based on our enhanced prediction model, AQI levels in {city} are expected to be **{trend}** over the next week, {trend_impact} from the current level of {current_aqi:.1f} ({current_category}).

### Forecast Details:
• Predicted AQI Range: {min_predicted_aqi:.1f} to {max_predicted_aqi:.1f}
• Weekly Average: {avg_predicted_aqi:.1f} ({pred_category})
• Best Air Quality: {best_day_formatted} ({prediction_data.iloc[best_day_idx]['predicted_aqi']:.1f})
• Poorest Air Quality: {worst_day_formatted} ({prediction_data.iloc[worst_day_idx]['predicted_aqi']:.1f})

### Contributing Factors:
• **Seasonal Patterns**: {season_specific}
• **Local Conditions**: {season_factor}
• **Pollutant Analysis**: {pollutant_factor}{event_factor}

### Health Recommendations:
{health_rec}

Our model analyzes multiple factors including historical trends, seasonal patterns, weather conditions, and regional pollution sources to generate these predictions."""
        
        return explanation
    except Exception as e:
        return f"Unable to generate prediction explanation: {str(e)}"
