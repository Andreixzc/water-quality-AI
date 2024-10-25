import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import unicodedata

def remove_accents(text):
    """Remove acentos e caracteres especiais do texto."""
    text = unicodedata.normalize('NFKD', text)
    text = text.encode('ASCII', 'ignore').decode('ASCII')
    return text

def sanitize_filename(name):
    """Sanitiza o nome do arquivo removendo caracteres especiais e acentos."""
    name = remove_accents(name)
    name = name.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
    return name.lower()  # Converter para minúsculas para maior consistência

def analyze_dataset():
    # Criar diretório para as figuras
    figures_dir = "figures"
    os.makedirs(figures_dir, exist_ok=True)

    # Configuração dos gráficos
    sns.set_theme(style="whitegrid")
    plt.rcParams['figure.figsize'] = [12, 6]

    # Ler o dataset
    data = pd.read_csv("../finished_processed_data/Base_kinross_filtered_parameters_updated.csv")

    # Definir parâmetros para análise
    exclude_columns = ["PONTO", "DATA", "Latitude", "Longitude"]
    parameters = [col for col in data.columns if col not in exclude_columns]

    # Gerar estatísticas descritivas
    desc_stats = data[parameters].describe()
    desc_stats.to_csv(os.path.join(figures_dir, 'descriptive_statistics.csv'))

    # Analisar cada parâmetro
    for param in parameters:
        print(f"Processando: {param}")
        safe_name = sanitize_filename(param)
        param_display = remove_accents(param)  # Para títulos dos gráficos

        # Distribuição
        plt.figure()
        sns.histplot(data[param], kde=True, bins=30)
        plt.title(f"Distribuicao de {param_display}", pad=20, fontsize=12)
        plt.xlabel(param_display, fontsize=10)
        plt.ylabel("Frequencia", fontsize=10)
        plt.savefig(os.path.join(figures_dir, f"dist_{safe_name}.png"), 
                    bbox_inches='tight', dpi=300)
        plt.close()

        # Boxplot
        plt.figure(figsize=(14, 6))
        sns.boxplot(x="PONTO", y=param, data=data)
        plt.title(f"Boxplot de {param_display} por Ponto de Amostragem", pad=20, fontsize=12)
        plt.xlabel("Ponto de Amostragem", fontsize=10)
        plt.ylabel(param_display, fontsize=10)
        plt.xticks(rotation=45)
        plt.savefig(os.path.join(figures_dir, f"box_{safe_name}.png"), 
                    bbox_inches='tight', dpi=300)
        plt.close()

        # Salvar estatísticas por ponto de amostragem
        point_stats = data.groupby("PONTO")[param].describe()
        point_stats.to_csv(os.path.join(figures_dir, f"{safe_name}_by_point.csv"))

    # Matriz de correlação
    plt.figure(figsize=(12, 10))
    corr_matrix = data[parameters].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, 
                mask=mask,
                annot=True, 
                cmap="coolwarm", 
                fmt=".2f",
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": .5})
    plt.title("Matriz de Correlacao entre Parametros", pad=20, fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "correlation_matrix.png"), 
                bbox_inches='tight', dpi=300)
    plt.close()

    # Salvar matriz de correlação em CSV
    corr_matrix.to_csv(os.path.join(figures_dir, 'correlation_matrix.csv'))

    # Salvar informações básicas do dataset
    dataset_info = {
        'total_samples': len(data),
        'total_columns': len(data.columns),
        'date_range': [data['DATA'].min(), data['DATA'].max()],
        'sampling_points': len(data['PONTO'].unique()),
        'parameters': len(parameters)
    }
    
    pd.DataFrame([dataset_info]).to_csv(os.path.join(figures_dir, 'dataset_info.csv'))

if __name__ == "__main__":
    analyze_dataset()