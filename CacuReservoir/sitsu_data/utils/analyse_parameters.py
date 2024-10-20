import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create a directory to store the analysis results
output_dir = "analysis_results"
os.makedirs(output_dir, exist_ok=True)

# Read the dataset
data = pd.read_csv("../finished_processed_data/Base_kinross_filtered_parameters_updated.csv")

# Save the first few rows of the dataset to a file
with open(os.path.join(output_dir, "dataset_head.txt"), "w") as file:
    file.write("First few rows of the dataset:\n")
    file.write(data.head().to_string(index=False))

# Get the column names
columns = data.columns.tolist()

# Exclude non-parameter columns
exclude_columns = ["PONTO", "DATA", "Latitude", "Longitude"]
parameters = [col for col in columns if col not in exclude_columns]

# Analyze each parameter
for param in parameters:
    # Create a directory for each parameter
    param_dir = os.path.join(output_dir, param)
    os.makedirs(param_dir, exist_ok=True)
    
    # Check for missing values and save to a file
    missing_values = data[param].isnull().sum()
    with open(os.path.join(param_dir, "missing_values.txt"), "w") as file:
        file.write(f"Missing values: {missing_values}\n")
    
    # Calculate total average and save to a file
    total_avg = data[param].mean()
    with open(os.path.join(param_dir, "total_average.txt"), "w") as file:
        file.write(f"Total average: {total_avg:.2f}\n")
    
    # Calculate average by sampling point and save to a file
    avg_by_point = data.groupby("PONTO")[param].mean()
    avg_by_point.to_csv(os.path.join(param_dir, "average_by_point.csv"), index=True)
    
    # Plot distribution and save to a file
    plt.figure(figsize=(8, 6))
    sns.histplot(data[param], kde=True)
    plt.title(f"Distribution of {param}")
    plt.xlabel(param)
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(param_dir, "distribution.png"))
    plt.close()
    
    # Plot boxplot by sampling point and save to a file
    plt.figure(figsize=(12, 6))
    sns.boxplot(x="PONTO", y=param, data=data)
    plt.title(f"Boxplot of {param} by Sampling Point")
    plt.xlabel("Sampling Point")
    plt.ylabel(param)
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(param_dir, "boxplot.png"))
    plt.close()

# Correlation matrix
corr_matrix = data[parameters].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.savefig(os.path.join(output_dir, "correlation_matrix.png"))
plt.close()