import pandas as pd


file_path = '../Base_kinross_all_parameters.csv'


df = pd.read_csv(file_path)


colunas_para_manter = [
    'PONTO', 
    'DATA', 
    'Clorofila a', 
    'Transparência da Água', 
    'Sólidos Dissolvidos', 
    'Sólidos Dissolvidos Totais', 
    'Sólidos Sedimentáveis', 
    'Sólidos Suspensos Totais', 
    'Sólidos Totais',
    'Turbidez'
]


df_filtrado = df[colunas_para_manter]


output_file = '../Base_kinross_filtered_parameters.csv'
df_filtrado.to_csv(output_file, index=False)

print(f"Arquivo filtrado salvo como '{output_file}'.")
