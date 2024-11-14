import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
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
        
        # Initialize calibration models
        self.calibration_models = {}
        
        # Define units for each parameter
        self.units = {
            'Clorofila a': 'μg/L',
            'Turbidez': 'NTU',
            'Transparência da Água': 'm',
            'Sólidos Dissolvidos Totais': 'mg/L'
        }
        
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
    
    def calibrate_indices(self, training_data, parameter):
        """
        Calibrate spectral indices using ground truth data
        Returns a fitted regression model for converting indices to actual units
        """
        # Calculate the index values
        index_values = training_data.apply(
            lambda row: self._calculate_raw_index(row, parameter), 
            axis=1
        )
        
        # Get actual measured values
        actual_values = training_data[parameter]
        
        # Fit linear regression model
        calibration_model = LinearRegression()
        calibration_model.fit(index_values.values.reshape(-1, 1), actual_values)
        
        # Store the calibration model
        self.calibration_models[parameter] = calibration_model
        
        # Calculate and log calibration metrics
        r2 = r2_score(actual_values, calibration_model.predict(index_values.values.reshape(-1, 1)))
        self.logger.info(f"Calibration R² for {parameter}: {r2:.3f}")
        
        return calibration_model
    
    def _calculate_raw_index(self, row, parameter):
        """Calculate raw spectral indices without calibration"""
        if parameter == 'Clorofila a':
            return (1/row['B3'] - 1/row['B4']) * row['B8']
        elif parameter == 'Turbidez':
            return (row['B4'] - row['B3']) / (row['B4'] + row['B3'])
        elif parameter == 'Transparência da Água':
            return row['B2'] / row['B4']
        elif parameter == 'Sólidos Dissolvidos Totais':
            return (row['B4'] - row['B3']) / (row['B4'] + row['B3'])
        else:
            raise ValueError(f"Unknown parameter: {parameter}")
    
    def calculate_chlorophyll_a(self, row):
        """Calculate Chlorophyll-a and calibrate to μg/L"""
        index = (1/row['B3'] - 1/row['B4']) * row['B8']
        if 'Clorofila a' in self.calibration_models:
            return self.calibration_models['Clorofila a'].predict([[index]])[0]
        return index
    
    def calculate_turbidity(self, row):
        """Calculate Turbidity and calibrate to NTU"""
        index = (row['B4'] - row['B3']) / (row['B4'] + row['B3'])
        if 'Turbidez' in self.calibration_models:
            return self.calibration_models['Turbidez'].predict([[index]])[0]
        return index
    
    def calculate_transparency(self, row):
        """Calculate water transparency and calibrate to meters"""
        index = row['B2'] / row['B4']
        if 'Transparência da Água' in self.calibration_models:
            return self.calibration_models['Transparência da Água'].predict([[index]])[0]
        return index
    
    def calculate_tss(self, row):
        """Calculate Total Suspended Solids and calibrate to mg/L"""
        index = (row['B4'] - row['B3']) / (row['B4'] + row['B3'])
        if 'Sólidos Dissolvidos Totais' in self.calibration_models:
            return self.calibration_models['Sólidos Dissolvidos Totais'].predict([[index]])[0]
        return index
    
    def calculate_parameter(self, row, parameter):
        """Calculate parameter based on spectral indices with calibration"""
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
        """Validate a single water quality parameter using all methods"""
        self.logger.info(f"Processing parameter: {parameter}")
        
        # Preprocess data
        processed_df = self.preprocess_data(df, parameter)
        
        # Split the data
        train_df, test_df = train_test_split(processed_df, test_size=test_size, random_state=42)
        
        # Calibrate indices using training data
        self.calibrate_indices(train_df, parameter)
        
        # Prepare features for ML model
        X_train = train_df[self.feature_columns]
        y_train = train_df[parameter]
        X_test = test_df[self.feature_columns]
        y_test = test_df[parameter]
        
        # Scale features for ML model
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Get ML predictions
        ml_predictions = model.predict(X_test_scaled)
        
        # Calculate calibrated indices-based estimations
        manual_predictions = X_test.apply(
            lambda row: self.calculate_parameter(row, parameter), 
            axis=1
        )
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'Actual': y_test,
            'ML_Predicted': ml_predictions,
            'Index_Based': manual_predictions
        })
        
        # Add units to column names if available
        if parameter in self.units:
            unit = self.units[parameter]
            results_df = results_df.rename(columns={
                'Actual': f'Actual ({unit})',
                'ML_Predicted': f'ML_Predicted ({unit})',
                'Index_Based': f'Index_Based ({unit})'
            })
        
        # Add key features for reference
        for col in self.band_columns:
            results_df[f'{col}_value'] = X_test[col].values
        
        # Calculate metrics
        metrics = {
            'ML_Model': {
                'R2': r2_score(results_df.iloc[:, 0], results_df.iloc[:, 1]),
                'RMSE': np.sqrt(mean_squared_error(results_df.iloc[:, 0], results_df.iloc[:, 1]))
            },
            'Index_Based': {
                'R2': r2_score(results_df.iloc[:, 0], results_df.iloc[:, 2]),
                'RMSE': np.sqrt(mean_squared_error(results_df.iloc[:, 0], results_df.iloc[:, 2]))
            }
        }
        
        return results_df, metrics
    
    def plot_comparison(self, results_df, parameter, metrics):
        """Create comparison plots for all methods with units"""
        plt.figure(figsize=(15, 10))
        
        # Get the actual column name (with units if available)
        actual_col = results_df.columns[0]  # First column is always Actual
        ml_col = results_df.columns[1]      # Second column is ML_Predicted
        index_col = results_df.columns[2]   # Third column is Index_Based
        
        # Time series plot
        plt.subplot(2, 1, 1)
        plt.plot(range(len(results_df)), results_df[actual_col], 'b-', 
                label='Actual', alpha=0.7)
        plt.plot(range(len(results_df)), results_df[ml_col], 'r--', 
                label=f'ML Model (R²={metrics["ML_Model"]["R2"]:.3f})', alpha=0.7)
        plt.plot(range(len(results_df)), results_df[index_col], 'g--', 
                label=f'Index Based (R²={metrics["Index_Based"]["R2"]:.3f})', alpha=0.7)
        plt.title(f'{parameter} - Comparison of Methods')
        plt.xlabel('Sample Index')
        plt.ylabel(actual_col)
        plt.legend()
        plt.grid(True)
        
        # Scatter plot
        plt.subplot(2, 1, 2)
        plt.scatter(results_df[actual_col], results_df[ml_col], 
                   alpha=0.5, label='ML Model')
        plt.scatter(results_df[actual_col], results_df[index_col], 
                   alpha=0.5, label='Index Based')
        
        # Add 1:1 line
        min_val = min(results_df[actual_col].min(), 
                     results_df[[ml_col, index_col]].min().min())
        max_val = max(results_df[actual_col].max(), 
                     results_df[[ml_col, index_col]].max().max())
        plt.plot([min_val, max_val], [min_val, max_val], 
                'k--', label='1:1 Line')
        
        plt.xlabel(actual_col)
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