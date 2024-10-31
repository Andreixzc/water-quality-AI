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

class WaterQualityMultiModel:
    def __init__(self, model_dir='models_performance_analysis'):
        """
        Initialize the water quality model with multiple model types
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
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

    def preprocess_data(self, df, target_variable, cloud_threshold=10):
        """
        Preprocess the data for a specific target variable
        """
        # Filter based on cloud percentage
        df = df[df['Cloud_Percentage'] <= cloud_threshold].copy()
        
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
        X = df[self.feature_columns]
        y = df[target_variable]
        
        # Handle missing values in features
        imputer = SimpleImputer(strategy='median')
        X = pd.DataFrame(
            imputer.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        return X, y

    def train_models(self, X, y):
        """
        Train multiple models and compare their performance
        """
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        results = {}
        
        for model_name, model_info in self.models.items():
            try:
                # Initialize and train the model
                model = model_info['model'](**model_info['params'])
                model.fit(X_train_scaled, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test_scaled)
                
                # Calculate metrics
                metrics = {
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'mae': mean_absolute_error(y_test, y_pred),
                    'r2': r2_score(y_test, y_pred),
                    'cv_scores': cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
                }
                
                # Get feature importance if available
                if hasattr(model, 'feature_importances_'):
                    feature_importance = pd.DataFrame({
                        'feature': X.columns,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)
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
        Save all models and their results
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_path = self.model_dir / f"{target_variable.replace(' ', '_')}_{timestamp}"
        results_path.mkdir(parents=True, exist_ok=True)
        
        # Save scaler
        joblib.dump(scaler, results_path / 'scaler.joblib')
        
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
            
            # Save individual model
            model_path = results_path / model_name
            model_path.mkdir(parents=True, exist_ok=True)
            
            # Save model
            joblib.dump(model_data['model'], model_path / 'model.joblib')
            
            # Save feature importance if available
            if model_data['feature_importance'] is not None:
                model_data['feature_importance'].to_csv(model_path / 'feature_importance.csv', index=False)
        
        # Save comparison metrics
        pd.DataFrame(comparison_metrics).to_csv(results_path / 'model_comparison.csv', index=False)
        
        self.logger.info(f"Results saved to {results_path}")
        return results_path

    def plot_model_comparison(self, results, target_variable, save_path):
        """
        Create comparison plots for all models
        """
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
        comparison_df.plot(x='Model', y=['R2 Score', 'RMSE', 'MAE'], kind='bar', ax=ax1)
        ax1.set_title(f'Model Performance Comparison - {target_variable}')
        ax1.grid(True)
        
        # 2. Predicted vs Actual plots
        ax2 = fig.add_subplot(gs[1, 0])
        for model_name, model_data in results.items():
            ax2.scatter(model_data['predictions']['y_test'], 
                       model_data['predictions']['y_pred'], 
                       alpha=0.5, 
                       label=model_name)
        ax2.plot([min(model_data['predictions']['y_test']), 
                  max(model_data['predictions']['y_test'])],
                 [min(model_data['predictions']['y_test']), 
                  max(model_data['predictions']['y_test'])], 
                 'r--')
        ax2.set_xlabel('Actual Values')
        ax2.set_ylabel('Predicted Values')
        ax2.set_title('Predicted vs Actual Values')
        ax2.legend()
        ax2.grid(True)
        
        # 3. Feature Importance Comparison (for models that support it)
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
            sns.heatmap(pivot_df, annot=True, cmap='YlOrRd', ax=ax3)
            ax3.set_title('Feature Importance Comparison')
        
        # 4. Cross-validation scores distribution
        ax4 = fig.add_subplot(gs[2, :])
        cv_data = []
        for model_name, model_data in results.items():
            cv_scores = model_data['metrics']['cv_scores']
            cv_data.extend([{'Model': model_name, 'CV Score': score} for score in cv_scores])
        cv_df = pd.DataFrame(cv_data)
        sns.boxplot(x='Model', y='CV Score', data=cv_df, ax=ax4)
        ax4.set_title('Cross-validation Score Distribution')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path / 'model_comparison.png')
        plt.close()

def train_all_parameters(file_path, parameters, cloud_threshold=10):
    """
    Train models for multiple water quality parameters
    """
    model_manager = WaterQualityMultiModel()
    df = pd.read_csv(file_path)
    
    results = {}
    for parameter in parameters:
        try:
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
            
        except Exception as e:
            model_manager.logger.error(f"Error processing {parameter}: {str(e)}")
    
    return results

# Example usage
if __name__ == "__main__":
    # Set parameters
    file_path = "../../sitsu_data/finished_processed_data/kinross_bandas_com_tolerancia_treino.csv"
    parameters = [
        'Clorofila a',
        'Transparência da Água',
        'Sólidos Dissolvidos Totais',
        'Turbidez'
    ]
    # Train all models
    results = train_all_parameters(file_path, parameters)
    
    # Print summary
    print("\nTraining Summary:")
    for parameter, result in results.items():
        print(f"\n{parameter}:")
        print(f"Results saved to: {result['path']}")
        print("\nModel Performance:")
        for model_name, model_data in result['model_results'].items():
            print(f"\n{model_name}:")
            print(f"R2 Score: {model_data['metrics']['r2']:.4f}")
            print(f"RMSE: {model_data['metrics']['rmse']:.4f}")
            print(f"MAE: {model_data['metrics']['mae']:.4f}")
            print(f"CV Score: {model_data['metrics']['cv_scores'].mean():.4f} "
                  f"(+/- {model_data['metrics']['cv_scores'].std()*2:.4f})")