import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
import joblib
from pathlib import Path
import logging
from datetime import datetime
import json
import os

class WaterQualityModel:
    def __init__(self, model_dir='tres_marias_models'):
        """
        Initialize the water quality model with multiple model types
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{model_dir}/model_training.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Define feature groups
        self.band_columns = ['B2', 'B3', 'B4', 'B5', 'B8', 'B11']
        self.index_columns = ['NDCI', 'NDVI', 'FAI', 'MNDWI', 'B3_B2_ratio', 'B4_B3_ratio', 'B5_B4_ratio']
        self.temporal_columns = ['Month', 'Season']
        self.feature_columns = self.band_columns + self.index_columns + self.temporal_columns

        # Define available models with their default parameters
        self.models = {
            'RandomForest': {
                'model': RandomForestRegressor,
                'params': {
                    'n_estimators': 100,
                    'max_depth': None,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                    'random_state': 42
                }
            },
            'GradientBoosting': {
                'model': GradientBoostingRegressor,
                'params': {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 3,
                    'random_state': 42
                }
            },
            'Lasso': {
                'model': LassoCV,
                'params': {
                    'cv': 5,
                    'random_state': 42
                }
            },
            'Ridge': {
                'model': RidgeCV,
                'params': {
                    'cv': 5
                }
            },
            'SVR': {
                'model': SVR,
                'params': {
                    'kernel': 'rbf',
                    'C': 1.0,
                    'epsilon': 0.1
                }
            }
        }

    def preprocess_data(self, df, target_variable, cloud_threshold=20):
        """
        Preprocess the data for a specific target variable
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
        
        # Filter based on cloud percentage if column exists
        if 'Cloud_Percentage' in df_processed.columns:
            initial_size = len(df_processed)
            df_processed = df_processed[df_processed['Cloud_Percentage'] <= cloud_threshold].copy()
            self.logger.info(f"Removed {initial_size - len(df_processed)} samples with cloud percentage > {cloud_threshold}%")
        else:
            self.logger.warning("Cloud_Percentage column not found, skipping cloud filtering")
        
        # Add Season column if Month exists but Season doesn't
        if 'Month' in df_processed.columns and 'Season' not in df_processed.columns:
            self.logger.info("Creating Season column from Month")
            df_processed['Season'] = (df_processed['Month'] % 12 + 3) // 3
        
        # Convert Season to numerical if it's categorical
        if 'Season' in df_processed.columns and df_processed['Season'].dtype == 'object':
            df_processed['Season'] = pd.Categorical(df_processed['Season']).codes
            self.logger.info("Converted Season column to numerical codes")
        
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
        self.logger.info(f"{target_variable} statistics: min={df_processed[target_variable].min()}, max={df_processed[target_variable].max()}, mean={df_processed[target_variable].mean()}, median={df_processed[target_variable].median()}")
        
        # Prepare features and target
        X = df_processed[self.feature_columns].copy()
        y = df_processed[target_variable].copy()
        
        # Handle missing values in features
        for col in X.columns:
            missing_count = X[col].isna().sum()
            if missing_count > 0:
                self.logger.info(f"Column {col} has {missing_count} missing values ({missing_count/len(X)*100:.1f}%)")
        
        # Use SimpleImputer to fill missing values
        imputer = SimpleImputer(strategy='median')
        X_imputed = pd.DataFrame(
            imputer.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        self.logger.info(f"Preprocessed dataset size: {len(X_imputed)} samples with {len(X_imputed.columns)} features")
        return X_imputed, y

    def train_models(self, X, y):
        """
        Train multiple models and compare their performance
        """
        self.logger.info(f"Training models on {len(X)} samples with {len(X.columns)} features")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.logger.info(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        results = {}
        
        for model_name, model_info in self.models.items():
            self.logger.info(f"Training {model_name} model...")
            try:
                # Initialize and train the model
                model = model_info['model'](**model_info['params'])
                model.fit(X_train_scaled, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test_scaled)
                
                # Calculate metrics
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
                
                metrics = {
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2,
                    'cv_scores': cv_scores
                }
                
                self.logger.info(f"{model_name} Performance - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}, CV R²: {cv_scores.mean():.4f} (±{cv_scores.std()*2:.4f})")
                
                # Get feature importance if available
                if hasattr(model, 'feature_importances_'):
                    feature_importance = pd.DataFrame({
                        'feature': X.columns,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    self.logger.info(f"Top 5 important features for {model_name}:")
                    for i, (_, row) in enumerate(feature_importance.head(5).iterrows()):
                        self.logger.info(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")
                else:
                    feature_importance = None
                
                results[model_name] = {
                    'model': model,
                    'metrics': metrics,
                    'feature_importance': feature_importance,
                    'predictions': {
                        'y_test': y_test,
                        'y_pred': y_pred
                    }
                }
                
            except Exception as e:
                self.logger.error(f"Error training {model_name}: {str(e)}")
        
        return results, scaler

    def save_results(self, results, scaler, target_variable):
        """
        Save all models and their results with parameter-specific names
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # Clean target variable name for file naming
        clean_target = target_variable.replace(' ', '_').replace('á', 'a').lower()
        results_path = self.model_dir / f"{clean_target}_{timestamp}"
        results_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Saving results to {results_path}")
        
        # Save scaler with parameter name
        scaler_filename = f'scaler_{clean_target}.joblib'
        joblib.dump(scaler, results_path / scaler_filename)
        
        # Create comparison metrics
        comparison_metrics = []
        for model_name, model_data in results.items():
            metrics = {
                'model': model_name,
                'rmse': model_data['metrics']['rmse'],
                'mae': model_data['metrics']['mae'],
                'r2': model_data['metrics']['r2'],
                'cv_mean': model_data['metrics']['cv_scores'].mean(),
                'cv_std': model_data['metrics']['cv_scores'].std()
            }
            comparison_metrics.append(metrics)
            
            # Save individual model with parameter name
            model_path = results_path / model_name
            model_path.mkdir(parents=True, exist_ok=True)
            
            # Create model filename with parameter name
            model_filename = f'{clean_target}_{model_name.lower()}_model.joblib'
            joblib.dump(model_data['model'], model_path / model_filename)
            
            # Save feature importance if available
            if model_data['feature_importance'] is not None:
                importance_filename = f'{clean_target}_{model_name.lower()}_feature_importance.csv'
                model_data['feature_importance'].to_csv(model_path / importance_filename, index=False)
        
        # Save comparison metrics with parameter name
        comparison_filename = f'{clean_target}_model_comparison.csv'
        pd.DataFrame(comparison_metrics).to_csv(results_path / comparison_filename, index=False)
        
        self.logger.info(f"Results saved to {results_path}")
        return results_path

    def plot_model_comparison(self, results, target_variable, save_path):
        """
        Create comparison plots for all models
        """
        self.logger.info(f"Creating comparison plots for {target_variable}")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(3, 2)
        
        # 1. Model Performance Comparison
        ax1 = fig.add_subplot(gs[0, :])
        comparison_data = []
        for model_name, model_data in results.items():
            comparison_data.append({
                'Model': model_name,
                'R2 Score': model_data['metrics']['r2'],
                'RMSE': model_data['metrics']['rmse'],
                'MAE': model_data['metrics']['mae']
            })
        comparison_df = pd.DataFrame(comparison_data)
        
        # Melt the dataframe for better plotting
        melted_df = pd.melt(comparison_df, id_vars=['Model'], 
                           value_vars=['R2 Score', 'RMSE', 'MAE'],
                           var_name='Metric', value_name='Value')
        
        # Create two separate plots for R2 and error metrics
        r2_df = melted_df[melted_df['Metric'] == 'R2 Score']
        error_df = melted_df[melted_df['Metric'] != 'R2 Score']
        
        # Plot R2 scores
        sns.barplot(x='Model', y='Value', data=r2_df, ax=ax1, color='green', alpha=0.7)
        ax1.set_title(f'Model R² Score Comparison - {target_variable}', fontsize=14)
        ax1.set_ylabel('R² Score', fontsize=12)
        ax1.set_ylim(0, 1)  # R2 typically between 0 and 1
        ax1.grid(True, axis='y')
        
        # Create a twin axis for error metrics
        ax1_twin = ax1.twinx()
        sns.barplot(x='Model', y='Value', hue='Metric', data=error_df, ax=ax1_twin, alpha=0.5)
        ax1_twin.set_ylabel('Error Value (RMSE, MAE)', fontsize=12)
        ax1_twin.legend(loc='upper right')
        
        # 2. Predicted vs Actual plots
        ax2 = fig.add_subplot(gs[1, 0])
        colors = ['blue', 'green', 'red', 'purple', 'orange']
        
        for i, (model_name, model_data) in enumerate(results.items()):
            ax2.scatter(model_data['predictions']['y_test'], 
                      model_data['predictions']['y_pred'], 
                      alpha=0.5, 
                      label=model_name,
                      color=colors[i % len(colors)])
        
        # Get min and max values for the identity line
        all_y_test = np.concatenate([model_data['predictions']['y_test'] for model_data in results.values()])
        all_y_pred = np.concatenate([model_data['predictions']['y_pred'] for model_data in results.values()])
        min_val = min(np.min(all_y_test), np.min(all_y_pred))
        max_val = max(np.max(all_y_test), np.max(all_y_pred))
        
        # Plot identity line
        ax2.plot([min_val, max_val], [min_val, max_val], 'k--')
        ax2.set_xlabel('Actual Values', fontsize=12)
        ax2.set_ylabel('Predicted Values', fontsize=12)
        ax2.set_title(f'Predicted vs Actual Values - {target_variable}', fontsize=14)
        ax2.legend()
        ax2.grid(True)
        
        # 3. Feature Importance Comparison
        ax3 = fig.add_subplot(gs[1, 1])
        feature_importance_data = []
        for model_name, model_data in results.items():
            if model_data['feature_importance'] is not None:
                top_features = model_data['feature_importance'].head(10)
                for _, row in top_features.iterrows():
                    feature_importance_data.append({
                        'Model': model_name,
                        'Feature': row['feature'],
                        'Importance': row['importance']
                    })
        
        if feature_importance_data:
            importance_df = pd.DataFrame(feature_importance_data)
            pivot_df = importance_df.pivot(index='Feature', 
                                        columns='Model', 
                                        values='Importance')
            # Fill NaN values for heatmap
            pivot_df = pivot_df.fillna(0)
            
            # Sort by average importance
            pivot_df['mean'] = pivot_df.mean(axis=1)
            pivot_df = pivot_df.sort_values('mean', ascending=False)
            pivot_df = pivot_df.drop('mean', axis=1)
            
            sns.heatmap(pivot_df, annot=True, cmap='YlGnBu', ax=ax3, fmt='.3f')
            ax3.set_title(f'Feature Importance Comparison - {target_variable}', fontsize=14)
        else:
            ax3.text(0.5, 0.5, 'Feature importance not available for these models', 
                   ha='center', va='center', fontsize=14)
        
        # 4. Cross-validation scores distribution
        ax4 = fig.add_subplot(gs[2, :])
        cv_data = []
        for model_name, model_data in results.items():
            cv_scores = model_data['metrics']['cv_scores']
            cv_data.extend([{'Model': model_name, 'CV Score': score} for score in cv_scores])
        cv_df = pd.DataFrame(cv_data)
        
        # Plot boxplot of CV scores
        sns.boxplot(x='Model', y='CV Score', data=cv_df, ax=ax4, palette='Set3')
        ax4.set_title(f'Cross-validation Score Distribution - {target_variable}', fontsize=14)
        ax4.set_ylabel('R² Score', fontsize=12)
        ax4.grid(True, axis='y')
        
        # Add mean and std as text
        for i, model_name in enumerate(results.keys()):
            model_cv = cv_df[cv_df['Model'] == model_name]['CV Score']
            ax4.text(i, model_cv.min() - 0.05, 
                   f'Mean: {model_cv.mean():.3f}\nStd: {model_cv.std():.3f}', 
                   ha='center', fontsize=10)
        
        # Save plot with parameter name
        clean_target = target_variable.replace(' ', '_').replace('á', 'a').lower()
        plt.tight_layout()
        plt.savefig(save_path / f'{clean_target}_model_comparison.png', dpi=300)
        plt.close()
        
        self.logger.info(f"Saved comparison plots to {save_path / f'{clean_target}_model_comparison.png'}")

def train_water_quality_models(file_path, cloud_threshold=20):
    """
    Train models for water quality parameters (turbidity and chlorophyll)
    """
    # Initialize model manager
    model_manager = WaterQualityModel()
    model_manager.logger.info(f"Loading data from {file_path}")
    
    # Load data
    try:
        df = pd.read_csv(file_path)
        model_manager.logger.info(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
    except Exception as e:
        model_manager.logger.error(f"Error loading data: {str(e)}")
        return None
    
    # Check if file exists
    if not os.path.exists(file_path):
        model_manager.logger.error(f"File not found: {file_path}")
        return None
    
    # Display dataset info
    model_manager.logger.info(f"Dataset columns: {df.columns.tolist()}")
    model_manager.logger.info(f"Dataset shape: {df.shape}")
    
    # Define target parameters (detect from dataset)
    target_parameters = []
    
    # Look for turbidity column (different possible names)
    turbidity_columns = ['turbidez', 'Turbidez', 'turbidity']
    found_turbidity = False
    for col in turbidity_columns:
        if col in df.columns:
            target_parameters.append(col)
            found_turbidity = True
            break
    
    if not found_turbidity:
        model_manager.logger.warning("No turbidity column found in dataset")
    
    # Look for chlorophyll column (different possible names)
    chlorophyll_columns = ['clorofila', 'Clorofila', 'clorofila_a', 'Clorofila_a', 'chlorophyll']
    found_chlorophyll = False
    for col in chlorophyll_columns:
        if col in df.columns:
            target_parameters.append(col)
            found_chlorophyll = True
            break
    
    if not found_chlorophyll:
        model_manager.logger.warning("No chlorophyll column found in dataset")
    
    if not target_parameters:
        model_manager.logger.error("No valid target parameters found in dataset")
        return None
    
    model_manager.logger.info(f"Training models for parameters: {target_parameters}")
    
    # Train models for each parameter
    results = {}
    for parameter in target_parameters:
        try:
            model_manager.logger.info(f"\n{'='*50}\nTraining models for {parameter}\n{'='*50}")
            
            # Preprocess data
            X, y = model_manager.preprocess_data(df, parameter, cloud_threshold)
            
            # Train all models
            model_results, scaler = model_manager.train_models(X, y)
            
            # Save results
            results_path = model_manager.save_results(model_results, scaler, parameter)
            
            # Create and save comparison plots
            model_manager.plot_model_comparison(model_results, parameter, results_path)
            
            # Store results
            results[parameter] = {
                'path': results_path,
                'model_results': model_results
            }
            
            model_manager.logger.info(f"Completed training for {parameter}")
            
        except Exception as e:
            model_manager.logger.error(f"Error processing {parameter}: {str(e)}")
    
    # Print summary
    model_manager.logger.info("\n\n" + "="*30 + " TRAINING SUMMARY " + "="*30)
    for parameter, result in results.items():
        model_manager.logger.info(f"\n{parameter}:")
        model_manager.logger.info(f"Results saved to: {result['path']}")
        model_manager.logger.info("\nModel Performance:")
        
        # Create a summary table
        summary_rows = []
        for model_name, model_data in result['model_results'].items():
            summary_rows.append({
                'Model': model_name,
                'R²': f"{model_data['metrics']['r2']:.4f}",
                'RMSE': f"{model_data['metrics']['rmse']:.4f}",
                'MAE': f"{model_data['metrics']['mae']:.4f}",
                'CV R²': f"{model_data['metrics']['cv_scores'].mean():.4f} ± {model_data['metrics']['cv_scores'].std()*2:.4f}"
            })
        
        # Find best model based on R2 score
        best_model = max(result['model_results'].items(), 
                       key=lambda x: x[1]['metrics']['r2'])
        
        model_manager.logger.info(f"Best model for {parameter}: {best_model[0]} (R² = {best_model[1]['metrics']['r2']:.4f})")
        
        # Create summary table
        summary_df = pd.DataFrame(summary_rows)
        model_manager.logger.info("\n" + summary_df.to_string(index=False))
    
    return results

if __name__ == "__main__":
    # Find most recent water quality with bands file
    data_dir = "./"
    filtered_files = [f for f in os.listdir(data_dir) if f.startswith("water_quality_with_bands_filtered_")]
    
    if filtered_files:
        # Sort by creation time (most recent first)
        filtered_files.sort(key=lambda x: os.path.getctime(os.path.join(data_dir, x)), reverse=True)
        latest_file = os.path.join(data_dir, filtered_files[0])
        print(f"Using most recent dataset: {latest_file}")
        
        # Train models
        results = train_water_quality_models(latest_file)
        
        if results:
            print("\nTraining completed successfully!")
        else:
            print("\nTraining failed. Check logs for details.")
    else:
        print("No water quality data files found. Please run the Earth Engine extraction script first.")