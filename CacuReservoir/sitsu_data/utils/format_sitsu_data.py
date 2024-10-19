import pandas as pd
import numpy as np

""""
Given the nature of the incorrectly formatted csv, the script below processes the data and formats the parameters from row to column.
"""
def process_csv(input_file, output_file):
    df = pd.read_csv(input_file)

    processed_data = {}

    current_point = None
    current_date = None
    row_data = {}

    for _, row in df.iterrows():
        if row['PONTO'] != current_point or row['DATA'] != current_date:
            if current_point is not None:
                key = (current_point, current_date)
                processed_data[key] = row_data
            current_point = row['PONTO']
            current_date = row['DATA']
            row_data = {'PONTO': current_point, 'DATA': current_date}
        
        row_data[row['Parâmetro']] = row['RESULT.']
        row_data[f"{row['Parâmetro']}_UN"] = row['UN.']

    if current_point is not None:
        key = (current_point, current_date)
        processed_data[key] = row_data

    new_df = pd.DataFrame.from_dict(processed_data, orient='index')
    new_df.reset_index(drop=True, inplace=True)
    columns = ['PONTO', 'DATA'] + sorted([col for col in new_df.columns if col not in ['PONTO', 'DATA']])
    new_df = new_df[columns]

    new_df.to_csv(output_file, index=False)

    print(f"Processed file saved:  {output_file}")

input_file = '../Base_kinross.csv'
output_file = '../Base_kinross_processed.csv'
process_csv(input_file, output_file)