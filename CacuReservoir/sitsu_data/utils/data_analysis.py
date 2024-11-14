import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

# Configuração para exibir todos os decimais
pd.set_option('display.float_format', lambda x: '%.3f' % x)

def create_output_dir(base_dir='analise_agua'):
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    return base_dir

def analyze_parameter(df, parameter, output_dir):
    print(f"\nAnalisando parâmetro: {parameter}")
    
    # Criar diretório específico para o parâmetro
    param_dir = os.path.join(output_dir, parameter.replace(" ", "_"))
    Path(param_dir).mkdir(exist_ok=True)
    
    # 1. Boxplot por ponto (mostra distribuição e outliers)
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='PONTO', y=parameter)
    plt.title(f'Distribuição de {parameter} por Ponto')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(param_dir, f'{parameter}_distribuicao_por_ponto.png'))
    plt.close()
    
    # 2. Calcular média por ponto
    medias = df.groupby('PONTO')[parameter].agg(['mean', 'count']).round(3)
    medias.columns = ['Média', 'Número de Amostras']
    medias.to_csv(os.path.join(param_dir, f'{parameter}_media_por_ponto.csv'))
    
    # 3. Identificar outliers
    Q1 = df[parameter].quantile(0.25)
    Q3 = df[parameter].quantile(0.75)
    IQR = Q3 - Q1
    outliers_mask = ((df[parameter] < (Q1 - 1.5 * IQR)) | 
                     (df[parameter] > (Q3 + 1.5 * IQR)))
    
    outliers = df[outliers_mask][['PONTO', 'DATA', parameter]]
    if not outliers.empty:
        outliers.to_csv(os.path.join(param_dir, f'{parameter}_outliers.csv'))

def main():
    # Criar diretório de saída
    output_dir = create_output_dir()
    
    # Ler dados
    filepath = '../finished_processed_data/Base_kinross_filtered_parameters_updated.csv'
    df = pd.read_csv(filepath, parse_dates=['DATA'])
    
    # Análise de valores ausentes
    missing_data = pd.DataFrame({
        'Valores Ausentes': df.isnull().sum(),
        'Porcentagem (%)': (df.isnull().sum() / len(df) * 100).round(2)
    })
    missing_data.to_csv(os.path.join(output_dir, 'valores_ausentes.csv'))
    
    # Lista de parâmetros para análise
    parameters = ['Clorofila a', 'Transparência da Água', 'Sólidos Dissolvidos',
                 'Sólidos Dissolvidos Totais', 'Sólidos Sedimentáveis',
                 'Sólidos Suspensos Totais', 'Sólidos Totais', 'Turbidez']
    
    # Análise para cada parâmetro
    for param in parameters:
        analyze_parameter(df, param, output_dir)

if __name__ == "__main__":
    main()