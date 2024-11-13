import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from datetime import datetime

class WaterQualityValidator:
    def __init__(self, output_dir='validation_results'):
        """Initialize the validator with output directory"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Define feature columns
        self.band_columns = ['B2', 'B3', 'B4', 'B5', 'B8', 'B11']
        self.index_columns = ['NDCI', 'NDVI', 'FAI', 'MNDWI', 'B3_B2_ratio', 'B4_B3_ratio', 'B5_B4_ratio']
        self.temporal_columns = ['Month', 'Season']
        self.feature_columns = self.band_columns + self.index_columns + self.temporal_columns
        
    def preprocess_data(self, df, target_variable, cloud_threshold=10):
        """Preprocess the data following the original script's method"""
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
        
        # Handle missing values in features
        imputer = SimpleImputer(strategy='median')
        feature_data = df[self.feature_columns]
        imputed_features = pd.DataFrame(
            imputer.fit_transform(feature_data),
            columns=feature_data.columns,
            index=feature_data.index
        )
        
        # Update the dataframe with imputed values
        for col in self.feature_columns:
            df[col] = imputed_features[col]
        
        return df
    
    def calculate_chlorophyll_a(self, row):
        """
        Calculate Chlorophyll-a using multiple indices
        Returns the 2BDA (Two Band Difference Algorithm) index
        """
        return (1/row['B3'] - 1/row['B4']) * row['B8']
    
    def calculate_turbidity(self, row):
        ## return NDTI index:
        return (row['B4'] - row['B3']) / (row['B4'] + row['B3'])
    
    def calculate_transparency(self, row):
        """
        Calculate water transparency using Blue/Red ratio
        """
        return row['B2'] / row['B4']
    
    def calculate_tss(self, row):
        """
        Calculate Total Suspended Solids using NDTI
        (Normalized Difference Turbidity Index)
        """
        return (row['B4'] - row['B3']) / (row['B4'] + row['B3'])
    
    def calculate_parameter(self, row, parameter):
        """Calculate parameter based on spectral indices"""
        if parameter == 'Clorofila a':
            return self.calculate_chlorophyll_a(row)
        elif parameter == 'Turbidez':
            return self.calculate_turbidity(row)
        elif parameter == 'Transparência da Água':
            return self.calculate_transparency(row)
        elif parameter == 'Sólidos Dissolvidos Totais':
            return self.calculate_tss(row)
        else:
            raise ValueError(f"Unknown parameter: {parameter}")
    
    def validate_parameter(self, df, parameter, test_size=0.2):
        """Validate a single water quality parameter using all three methods"""
        self.logger.info(f"Processing parameter: {parameter}")
        
        # Preprocess data
        processed_df = self.preprocess_data(df, parameter)
        
        # Prepare features
        X = processed_df[self.feature_columns]
        y = processed_df[parameter]
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features for ML model
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Get ML predictions
        ml_predictions = model.predict(X_test_scaled)
        
        # Calculate indices-based estimations
        manual_predictions = X_test.apply(lambda row: self.calculate_parameter(row, parameter), axis=1)
        
        # Create results DataFrame with original features for reference
        results_df = pd.DataFrame({
            'Actual': y_test,
            'ML_Predicted': ml_predictions,
            'Index_Based': manual_predictions
        })
        
        # Add key features for reference
        for col in self.band_columns:
            results_df[f'{col}_value'] = X_test[col].values
        
        # Calculate metrics
        metrics = {
            'ML_Model': {
                'R2': r2_score(results_df['Actual'], results_df['ML_Predicted']),
                'RMSE': np.sqrt(mean_squared_error(results_df['Actual'], results_df['ML_Predicted']))
            },
            'Index_Based': {
                'R2': r2_score(results_df['Actual'], results_df['Index_Based']),
                'RMSE': np.sqrt(mean_squared_error(results_df['Actual'], results_df['Index_Based']))
            }
        }
        
        return results_df, metrics
    
    def plot_comparison(self, results_df, parameter, metrics):
        """Create comparison plots for all three methods"""
        plt.figure(figsize=(15, 10))
        
        # Time series plot
        plt.subplot(2, 1, 1)
        plt.plot(range(len(results_df)), results_df['Actual'], 'b-', label='Actual', alpha=0.7)
        plt.plot(range(len(results_df)), results_df['ML_Predicted'], 'r--', 
                label=f'ML Model (R²={metrics["ML_Model"]["R2"]:.3f})', alpha=0.7)
        plt.plot(range(len(results_df)), results_df['Index_Based'], 'g--', 
                label=f'Index Based (R²={metrics["Index_Based"]["R2"]:.3f})', alpha=0.7)
        plt.title(f'{parameter} - Comparison of Methods')
        plt.xlabel('Sample Index')
        plt.ylabel(parameter)
        plt.legend()
        plt.grid(True)
        
        # Scatter plot
        plt.subplot(2, 1, 2)
        plt.scatter(results_df['Actual'], results_df['ML_Predicted'], 
                   alpha=0.5, label='ML Model')
        plt.scatter(results_df['Actual'], results_df['Index_Based'], 
                   alpha=0.5, label='Index Based')
        plt.plot([results_df['Actual'].min(), results_df['Actual'].max()],
                [results_df['Actual'].min(), results_df['Actual'].max()],
                'k--', label='1:1 Line')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted/Calculated Values')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        return plt.gcf()
    
    def run_validation(self, data_path, parameters):
        """Run validation for all specified parameters"""
        # Read data
        df = pd.read_csv(data_path)
        
        # Process each parameter
        for parameter in parameters:
            try:
                # Run validation
                results_df, metrics = self.validate_parameter(df, parameter)
                
                # Save results
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                clean_param = parameter.replace(' ', '_').replace('á', 'a').lower()
                
                # Save CSV
                results_path = self.output_dir / f'{clean_param}_validation_{timestamp}.csv'
                results_df.to_csv(results_path, index=False)
                
                # Save metrics
                metrics_path = self.output_dir / f'{clean_param}_metrics_{timestamp}.csv'
                metrics_df = pd.DataFrame({
                    'Method': ['ML_Model', 'Index_Based'],
                    'R2': [metrics['ML_Model']['R2'], metrics['Index_Based']['R2']],
                    'RMSE': [metrics['ML_Model']['RMSE'], metrics['Index_Based']['RMSE']]
                })
                metrics_df.to_csv(metrics_path, index=False)
                
                # Create and save plot
                fig = self.plot_comparison(results_df, parameter, metrics)
                plot_path = self.output_dir / f'{clean_param}_plot_{timestamp}.png'
                fig.savefig(plot_path)
                plt.close()
                
                self.logger.info(f"Results for {parameter} saved to {self.output_dir}")
                
            except Exception as e:
                self.logger.error(f"Error processing {parameter}: {str(e)}")

# Example usage
if __name__ == "__main__":
    # Initialize validator
    validator = WaterQualityValidator()
    
    # Set parameters to validate
    parameters = [
        'Clorofila a',
        'Transparência da Água',
        'Sólidos Dissolvidos Totais',
        'Turbidez'
    ]
    
    # Run validation
    validator.run_validation(
        data_path="../../sitsu_data/finished_processed_data/kinross_bandas_com_tolerancia_treino.csv",
        parameters=parameters
    )