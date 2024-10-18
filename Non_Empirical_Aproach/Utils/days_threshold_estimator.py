import pandas as pd

csvPath = "../../sitsu_data/Base_kinross_processed.csv"
df = pd.read_csv(csvPath)
print(df.head())