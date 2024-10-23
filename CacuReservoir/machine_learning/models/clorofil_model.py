import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

def load_and_preprocess_data(file_path, target_variable='Clorofila a', cloud_threshold=10):
    """
    Load and preprocess the water quality data
    """
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Filter based on cloud percentage
    df = df[df['Cloud_Percentage'] <= cloud_threshold].copy()
    
    # Define feature columns
    band_columns = ['B2', 'B3', 'B4', 'B5', 'B8', 'B11']
    index_columns = ['NDCI', 'NDVI', 'FAI', 'MNDWI', 'B3_B2_ratio', 'B4_B3_ratio', 'B5_B4_ratio']
    temporal_columns = ['Month', 'Season']
    
    feature_columns = band_columns + index_columns + temporal_columns
    
    # Convert Season to numerical if it's categorical
    if df['Season'].dtype == 'object':
        df['Season'] = pd.Categorical(df['Season']).codes
    
    # Remove rows where target variable is missing
    df = df.dropna(subset=[target_variable])
    
    # Handle outliers using IQR method for target variable
    Q1 = df[target_variable].quantile(0.25)
    Q3 = df[target_variable].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Filter outliers
    df = df[(df[target_variable] >= lower_bound) & 
            (df[target_variable] <= upper_bound)]
    
    # Prepare features and target
    X = df[feature_columns]
    y = df[target_variable]
    
    # Handle missing values in features
    imputer = SimpleImputer(strategy='median')
    X = pd.DataFrame(
        imputer.fit_transform(X),
        columns=X.columns,
        index=X.index
    )
    
    print(f"Final dataset shape: {X.shape}")
    print(f"Number of samples for target variable: {len(y)}")
    
    return X, y

def train_model(X, y, test_size=0.2, random_state=42):
    """
    Train a Random Forest model and evaluate its performance
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame to preserve column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    # Initialize and train the model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=random_state
    )
    
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Perform cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return {
        'model': model,
        'scaler': scaler,
        'metrics': {
            'rmse': rmse,
            'r2': r2,
            'cv_scores_mean': cv_scores.mean(),
            'cv_scores_std': cv_scores.std()
        },
        'feature_importance': feature_importance,
        'predictions': {
            'y_test': y_test,
            'y_pred': y_pred
        }
    }

def plot_results(results):
    """
    Plot the model results and feature importance
    """
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot predicted vs actual values
    ax1.scatter(results['predictions']['y_test'], results['predictions']['y_pred'], alpha=0.5)
    ax1.plot([min(results['predictions']['y_test']), max(results['predictions']['y_test'])], 
             [min(results['predictions']['y_test']), max(results['predictions']['y_test'])], 
             'r--')
    ax1.set_xlabel('Actual Values')
    ax1.set_ylabel('Predicted Values')
    ax1.set_title('Predicted vs Actual Values')
    
    # Plot feature importance
    feature_importance = results['feature_importance'].head(10)  # Top 10 features
    ax2.barh(feature_importance['feature'], feature_importance['importance'])
    ax2.set_xlabel('Importance')
    ax2.set_title('Top 10 Feature Importance')
    
    plt.tight_layout()
    return fig

# Example usage
if __name__ == "__main__":
    # Set parameters
    file_path = "../../sitsu_data/finished_processed_data/kinross_bandas_com_tolerancia_treino.csv"
    target_variable = "Clorofila a"
    cloud_threshold = 10
    
    # Load and preprocess data
    X, y = load_and_preprocess_data(file_path, target_variable, cloud_threshold)
    
    # Train and evaluate model
    results = train_model(X, y)
    
    # Print metrics
    print("\nModel Performance Metrics:")
    print(f"RMSE: {results['metrics']['rmse']:.4f}")
    print(f"R2 Score: {results['metrics']['r2']:.4f}")
    print(f"Cross-validation R2 Score: {results['metrics']['cv_scores_mean']:.4f} (+/- {results['metrics']['cv_scores_std']*2:.4f})")
    
    # Plot results
    plot_results(results)
    plt.show()