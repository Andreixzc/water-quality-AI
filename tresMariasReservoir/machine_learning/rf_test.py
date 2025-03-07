import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from datetime import datetime
import joblib
import time

# Machine learning imports
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFECV, SelectFromModel
from sklearn.inspection import permutation_importance

class RandomForestOptimizer:
    def __init__(self, model_dir='rf_optimized_models'):
        """
        Initialize the Random Forest optimizer for water quality prediction
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{model_dir}/optimization.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Define feature groups
        self.band_columns = ['B2', 'B3', 'B4', 'B5', 'B8', 'B11']
        self.index_columns = ['NDCI', 'NDVI', 'FAI', 'MNDWI', 'B3_B2_ratio', 'B4_B3_ratio', 'B5_B4_ratio']
        self.temporal_columns = ['Month', 'Season']
        self.feature_columns = self.band_columns + self.index_columns + self.temporal_columns
        
        # Set random seed for reproducibility
        np.random.seed(42)
    
    def preprocess_data(self, df, target_variable, cloud_threshold=20):
        """
        Preprocess data with more advanced strategies
        """
        self.logger.info(f"Preprocessing data for {target_variable} with cloud threshold {cloud_threshold}%")
        
        # Make a copy to avoid modifying the original dataframe
        df_processed = df.copy()
        
        # Verify all required columns exist
        missing_features = [col for col in self.feature_columns if col not in df_processed.columns]
        if missing_features:
            self.logger.warning(f"Missing feature columns: {missing_features}")
            # Remove missing features from the feature list
            for col in missing_features:
                if col in self.band_columns:
                    self.band_columns.remove(col)
                elif col in self.index_columns:
                    self.index_columns.remove(col)
                elif col in self.temporal_columns:
                    self.temporal_columns.remove(col)
            # Update feature columns
            self.feature_columns = self.band_columns + self.index_columns + self.temporal_columns
        
        # Filter based on cloud percentage
        if 'Cloud_Percentage' in df_processed.columns:
            initial_size = len(df_processed)
            df_processed = df_processed[df_processed['Cloud_Percentage'] <= cloud_threshold].copy()
            self.logger.info(f"Removed {initial_size - len(df_processed)} samples with cloud percentage > {cloud_threshold}%")
        
        # Create Season if needed
        if 'Month' in df_processed.columns and 'Season' not in df_processed.columns:
            self.logger.info("Creating Season column from Month")
            df_processed['Season'] = (df_processed['Month'] % 12 + 3) // 3
        
        # Make sure Season is numerical
        if 'Season' in df_processed.columns and df_processed['Season'].dtype == 'object':
            df_processed['Season'] = pd.Categorical(df_processed['Season']).codes
        
        # Check if target variable exists
        if target_variable not in df_processed.columns:
            self.logger.error(f"Target variable '{target_variable}' not found in dataset")
            raise ValueError(f"Target variable '{target_variable}' not found in dataset")
        
        # Remove rows where target variable is missing
        initial_size = len(df_processed)
        df_processed = df_processed.dropna(subset=[target_variable])
        self.logger.info(f"Removed {initial_size - len(df_processed)} samples with missing {target_variable} values")
        
        # Handle outliers using IQR method for target variable
        Q1 = df_processed[target_variable].quantile(0.25)
        Q3 = df_processed[target_variable].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Filter outliers
        initial_size = len(df_processed)
        df_processed = df_processed[(df_processed[target_variable] >= lower_bound) & 
                (df_processed[target_variable] <= upper_bound)]
        self.logger.info(f"Removed {initial_size - len(df_processed)} outlier samples for {target_variable}")
        
        # Log target variable statistics
        self.logger.info(f"{target_variable} statistics: min={df_processed[target_variable].min():.4f}, " +
                       f"max={df_processed[target_variable].max():.4f}, " + 
                       f"mean={df_processed[target_variable].mean():.4f}, " +
                       f"median={df_processed[target_variable].median():.4f}")
        
        # Prepare features and target
        X = df_processed[self.feature_columns].copy()
        y = df_processed[target_variable].copy()
        
        # Enhanced feature engineering 
        # Add ratio of reflectance bands to help with water quality detection
        if 'B4' in X.columns and 'B5' in X.columns and 'B3' in X.columns:
            X['B4_B5_ratio'] = X['B4'] / (X['B5'] + 0.0001)  # Red to Red Edge ratio
            X['B3_B5_ratio'] = X['B3'] / (X['B5'] + 0.0001)  # Green to Red Edge ratio
        
        # Add square and cube of important indices (non-linear relationships)
        if 'NDCI' in X.columns:
            X['NDCI_squared'] = X['NDCI'] ** 2
        
        if 'MNDWI' in X.columns:
            X['MNDWI_squared'] = X['MNDWI'] ** 2
        
        # Update feature columns with new engineered features
        self.feature_columns = list(X.columns)
        
        # Handle missing values with more sophisticated approach
        self.logger.info("Imputing missing values...")
        
        # Log missing values by column
        missing_counts = X.isna().sum()
        missing_cols = missing_counts[missing_counts > 0]
        if len(missing_cols) > 0:
            self.logger.info("Missing values by column:")
            for col, count in missing_cols.items():
                self.logger.info(f"  {col}: {count} ({count/len(X)*100:.1f}%)")
        
        # Use median imputation
        imputer = SimpleImputer(strategy='median')
        X_imputed = pd.DataFrame(
            imputer.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        # Split data for train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X_imputed, y, test_size=0.2, random_state=42, shuffle=True
        )
        
        # Scale features - while tree-based models don't require scaling, 
        # it can help with feature importance interpretation
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        self.logger.info(f"Preprocessed dataset size: {len(X_imputed)} samples with {len(X_imputed.columns)} features")
        self.logger.info(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")
        
        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train, 
            'y_test': y_test,
            'scaler': scaler,
            'imputer': imputer,
            'feature_names': X.columns
        }
    
    def analyze_feature_importance(self, X_train, y_train, n_estimators=100):
        """
        Analyze feature importance using multiple methods
        """
        self.logger.info("Analyzing feature importance...")
        
        # Train a basic Random Forest model
        rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        rf.fit(X_train, y_train)
        
        # Get feature importance from the model
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Create a DataFrame with feature importances
        feature_importance_df = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        self.logger.info("Top 10 features by importance:")
        for i, (_, row) in enumerate(feature_importance_df.head(10).iterrows()):
            self.logger.info(f"  {i+1}. {row['Feature']}: {row['Importance']:.4f}")
        
        # Perform permutation importance (more reliable than built-in feature importance)
        self.logger.info("Calculating permutation importance (this may take a while)...")
        perm_importance = permutation_importance(rf, X_train, y_train, n_repeats=10, random_state=42)
        
        perm_importance_df = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': perm_importance.importances_mean,
            'Std': perm_importance.importances_std
        }).sort_values('Importance', ascending=False)
        
        self.logger.info("Top 10 features by permutation importance:")
        for i, (_, row) in enumerate(perm_importance_df.head(10).iterrows()):
            self.logger.info(f"  {i+1}. {row['Feature']}: {row['Importance']:.4f} ± {row['Std']:.4f}")
        
        return {
            'feature_importance': feature_importance_df,
            'permutation_importance': perm_importance_df
        }
    
    def select_best_features(self, X_train, y_train, X_test, importance_data, threshold=0.95):
        """
        Select best features based on cumulative importance
        """
        self.logger.info("Selecting best features...")
        
        # Get feature importance
        feature_importance = importance_data['permutation_importance']
        
        # Sort by importance
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        
        # Calculate cumulative importance
        feature_importance['Cumulative'] = feature_importance['Importance'].cumsum() / feature_importance['Importance'].sum()
        
        # Select features based on cumulative importance threshold
        selected_features = feature_importance[feature_importance['Cumulative'] <= threshold]['Feature'].tolist()
        
        # Make sure we have at least 5 features
        if len(selected_features) < 5:
            selected_features = feature_importance.head(5)['Feature'].tolist()
        
        self.logger.info(f"Selected {len(selected_features)} features out of {len(X_train.columns)}")
        self.logger.info(f"Selected features: {', '.join(selected_features)}")
        
        # Filter datasets to include only selected features
        X_train_selected = X_train[selected_features]
        X_test_selected = X_test[selected_features]
        
        return {
            'X_train': X_train_selected,
            'X_test': X_test_selected,
            'selected_features': selected_features
        }
    
    def optimize_random_forest(self, X_train, y_train, cv=5, n_iter=20):
        """
        Optimize RandomForest hyperparameters using RandomizedSearchCV
        """
        self.logger.info("Optimizing Random Forest hyperparameters...")
        
        # Define parameter grid for RandomizedSearchCV
        param_grid = {
            'n_estimators': [50, 100, 200, 300, 500],
            'max_depth': [None, 10, 20, 30, 40, 50],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt', 'log2', None],
            'bootstrap': [True, False]
        }
        
        # Initial RF model
        rf = RandomForestRegressor(random_state=42)
        
        # Use RandomizedSearchCV to find best parameters
        random_search = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv,
            verbose=1,
            random_state=42,
            n_jobs=-1,
            scoring='r2'
        )
        
        # Fit RandomizedSearchCV
        start_time = time.time()
        random_search.fit(X_train, y_train)
        search_time = time.time() - start_time
        
        # Get best parameters
        best_params = random_search.best_params_
        
        self.logger.info(f"RandomizedSearchCV completed in {search_time:.2f} seconds")
        self.logger.info(f"Best parameters: {best_params}")
        self.logger.info(f"Best CV score: {random_search.best_score_:.4f}")
        
        # Create model with best parameters
        best_rf = RandomForestRegressor(
            n_estimators=best_params['n_estimators'],
            max_depth=best_params['max_depth'],
            min_samples_split=best_params['min_samples_split'],
            min_samples_leaf=best_params['min_samples_leaf'],
            max_features=best_params['max_features'],
            bootstrap=best_params['bootstrap'],
            random_state=42
        )
        
        # Fit on the whole training set
        best_rf.fit(X_train, y_train)
        
        return {
            'model': best_rf,
            'best_params': best_params,
            'cv_score': random_search.best_score_
        }
    
    def evaluate_model(self, model, X_test, y_test):
        """
        Evaluate the model on test data
        """
        self.logger.info("Evaluating model on test data...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        self.logger.info(f"Test metrics - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        
        # Calculate error statistics
        errors = y_pred - y_test
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        
        self.logger.info(f"Error statistics - Mean: {mean_error:.4f}, Std: {std_error:.4f}")
        
        return {
            'predictions': y_pred,
            'metrics': {
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            },
            'errors': errors
        }
    
    def save_model(self, model, data, results, target_variable):
        """
        Save model, parameters, and results
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        clean_target = target_variable.replace(' ', '_').replace('á', 'a').lower()
        save_path = self.model_dir / f"optimized_rf_{clean_target}_{timestamp}"
        save_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Saving model and results to {save_path}")
        
        # Save model
        joblib.dump(model, save_path / f"{clean_target}_rf_model.joblib")
        
        # Save scaler and imputer
        joblib.dump(data['scaler'], save_path / f"{clean_target}_scaler.joblib")
        joblib.dump(data['imputer'], save_path / f"{clean_target}_imputer.joblib")
        
        # Save feature names
        with open(save_path / "feature_names.txt", "w") as f:
            f.write("\n".join(data['feature_names']))
        
        # Save selected features
        with open(save_path / "selected_features.txt", "w") as f:
            f.write("\n".join(results['selected_features']))
        
        # Save best parameters
        with open(save_path / "best_params.txt", "w") as f:
            for param, value in results['best_params'].items():
                f.write(f"{param}: {value}\n")
        
        # Save metrics
        metrics_df = pd.DataFrame([{
            'Target': target_variable,
            'R2': results['metrics']['r2'],
            'RMSE': results['metrics']['rmse'],
            'MAE': results['metrics']['mae'],
            'CV_R2': results['cv_score']
        }])
        metrics_df.to_csv(save_path / "metrics.csv", index=False)
        
        # Create visualizations
        self.create_visualizations(results, target_variable, save_path)
        
        return save_path
    
    def create_visualizations(self, results, target_variable, save_path):
        """
        Create visualizations of model results
        """
        self.logger.info("Creating visualizations...")
        
        # 1. Predicted vs Actual plot
        plt.figure(figsize=(10, 8))
        plt.scatter(results['y_test'], results['predictions'], alpha=0.5)
        plt.plot([results['y_test'].min(), results['y_test'].max()], 
                [results['y_test'].min(), results['y_test'].max()], 'r--')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Predicted vs Actual - {target_variable} (R² = {results["metrics"]["r2"]:.4f})')
        plt.grid(True)
        plt.savefig(save_path / "predicted_vs_actual.png", dpi=300)
        plt.close()
        
        # 2. Feature importance plot
        feature_importance = pd.DataFrame({
            'Feature': results['selected_features'],
            'Importance': results['model'].feature_importances_
        }).sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance)
        plt.title(f'Feature Importance - {target_variable}')
        plt.tight_layout()
        plt.savefig(save_path / "feature_importance.png", dpi=300)
        plt.close()
        
        # 3. Error distribution
        plt.figure(figsize=(10, 8))
        sns.histplot(results['errors'], kde=True)
        plt.axvline(x=0, color='r', linestyle='--')
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.title(f'Error Distribution - {target_variable}')
        plt.grid(True)
        plt.savefig(save_path / "error_distribution.png", dpi=300)
        plt.close()
        
        # 4. Sorted predictions
        plt.figure(figsize=(12, 6))
        
        # Sort test data by actual values
        sorted_indices = np.argsort(results['y_test'])
        sorted_y_test = np.array(results['y_test'])[sorted_indices]
        sorted_y_pred = np.array(results['predictions'])[sorted_indices]
        
        # Plot
        plt.plot(range(len(sorted_y_test)), sorted_y_test, 'b-', label='Actual')
        plt.plot(range(len(sorted_y_pred)), sorted_y_pred, 'r-', alpha=0.7, label='Predicted')
        plt.xlabel('Sorted Sample Index')
        plt.ylabel('Value')
        plt.title(f'Sorted Actual vs Predicted - {target_variable}')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path / "sorted_predictions.png", dpi=300)
        plt.close()

def optimize_water_quality_model(file_path, target_parameter, cloud_threshold=20):
    """
    Optimize a Random Forest model for water quality prediction
    """
    # Initialize optimizer
    optimizer = RandomForestOptimizer()
    optimizer.logger.info(f"Starting optimization for {target_parameter}")
    
    # Load data
    try:
        df = pd.read_csv(file_path)
        optimizer.logger.info(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
    except Exception as e:
        optimizer.logger.error(f"Error loading data: {str(e)}")
        return None
    
    # Preprocess data
    data = optimizer.preprocess_data(df, target_parameter, cloud_threshold)
    
    # Analyze feature importance
    importance_data = optimizer.analyze_feature_importance(data['X_train'], data['y_train'])
    
    # Select best features
    selected_data = optimizer.select_best_features(
        data['X_train'], data['y_train'], data['X_test'], importance_data
    )
    
    # Optimize Random Forest
    optimized_rf = optimizer.optimize_random_forest(
        selected_data['X_train'], data['y_train'], cv=5, n_iter=30
    )
    
    # Evaluate optimized model
    evaluation = optimizer.evaluate_model(
        optimized_rf['model'], selected_data['X_test'], data['y_test']
    )
    
    # Combine results
    results = {
        'model': optimized_rf['model'],
        'best_params': optimized_rf['best_params'],
        'cv_score': optimized_rf['cv_score'],
        'metrics': evaluation['metrics'],
        'predictions': evaluation['predictions'],
        'y_test': data['y_test'],
        'errors': evaluation['errors'],
        'selected_features': selected_data['selected_features'],
        'feature_importance': importance_data['feature_importance'],
        'permutation_importance': importance_data['permutation_importance']
    }
    
    # Save model and results
    save_path = optimizer.save_model(optimized_rf['model'], data, results, target_parameter)
    
    return {
        'model': optimized_rf['model'],
        'path': save_path,
        'metrics': evaluation['metrics'],
        'best_params': optimized_rf['best_params']
    }

if __name__ == "__main__":
    # Find most recent water quality with bands file
    data_dir = "./"
    filtered_files = [f for f in os.listdir(data_dir) if f.startswith("water_quality_with_bands_filtered_")]
    
    if filtered_files:
        # Sort by creation time (most recent first)
        filtered_files.sort(key=lambda x: os.path.getctime(os.path.join(data_dir, x)), reverse=True)
        latest_file = os.path.join(data_dir, filtered_files[0])
        print(f"Using most recent dataset: {latest_file}")
        
        # Define target parameters
        target_parameters = []
        
        # Load the file to check available parameters
        try:
            df = pd.read_csv(latest_file)
            # Check for turbidity
            if any(col in df.columns for col in ['turbidez', 'Turbidez']):
                turbidity_col = next(col for col in ['turbidez', 'Turbidez'] if col in df.columns)
                target_parameters.append(turbidity_col)
            
            # Check for chlorophyll
            if any(col in df.columns for col in ['clorofila', 'Clorofila']):
                chlorophyll_col = next(col for col in ['clorofila', 'Clorofila'] if col in df.columns)
                target_parameters.append(chlorophyll_col)
        except Exception as e:
            print(f"Error checking parameters: {str(e)}")
            # Default parameters if check fails
            target_parameters = ['turbidez', 'clorofila']
        
        # Optimize models for each parameter
        for param in target_parameters:
            print(f"\n{'='*50}\nOptimizing model for {param}\n{'='*50}")
            result = optimize_water_quality_model(latest_file, param)
            
            if result:
                print(f"\nOptimization completed for {param}:")
                print(f"Model saved to: {result['path']}")
                print(f"Performance metrics:")
                print(f"  R² score: {result['metrics']['r2']:.4f}")
                print(f"  RMSE: {result['metrics']['rmse']:.4f}")
                print(f"  MAE: {result['metrics']['mae']:.4f}")
                print("\nBest parameters:")
                for param_name, value in result['best_params'].items():
                    print(f"  {param_name}: {value}")
            else:
                print(f"Optimization failed for {param}. Check logs for details.")
    else:
        print("No water quality data files found. Please run the Earth Engine extraction script first.")