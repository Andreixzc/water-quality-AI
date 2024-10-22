# earth_engine/extracao_bandas.py

import pandas as pd
import ee

"""
Este script extrai bandas espectrais e calcula índices de vegetação/água usando o Google Earth Engine.
Ele processa dados de clorofila com coordenadas geográficas, busca imagens de satélite correspondentes,
e calcula vários índices espectrais úteis para a análise de qualidade da água.

Funcionalidade:
1. Inicializa o Google Earth Engine.
2. Carrega dados de clorofila com coordenadas geográficas.
3. Para cada ponto de amostra, busca a imagem de satélite mais próxima dentro de um intervalo de tempo.
4. Extrai valores de bandas espectrais e calcula índices relevantes.
5. Adiciona informações temporais (mês e estação) aos dados.
6. Salva os dados processados em arquivos CSV para treinamento de modelos, incluindo apenas instâncias com imagens encontradas.
"""

# Inicializar o Google Earth Engine
ee.Initialize()

# Definir a tolerância de dias como uma variável parametrizável
DIAS_TOLERANCIA = 30

# Ler o arquivo CSV com dados de clorofila e coordenadas
df = pd.read_csv("../../sitsu_data/finished_processed_data/Base_kinross_filtered_parameters_updated.csv")

# Converter a coluna 'DATA' para o formato de data
df['DATA'] = pd.to_datetime(df['DATA'])

def extract_bands_and_indices(row):
    """
    Função para extrair bandas espectrais e calcular índices para um ponto específico.
    
    Args:
        row (pd.Series): Uma linha do DataFrame contendo informações de um ponto de amostra.
    
    Returns:
        pd.Series: Série com valores das bandas e índices calculados, ou pd.Series com valores nulos se nenhuma imagem for encontrada.
    """
    point = ee.Geometry.Point(row['Longitude'], row['Latitude'])
    date = ee.Date(row['DATA'].strftime('%Y-%m-%d'))

    # Definir o intervalo de datas para busca
    start_date = date.advance(-DIAS_TOLERANCIA, 'day')
    end_date = date.advance(DIAS_TOLERANCIA, 'day')

    # Buscar a coleção de imagens Sentinel-2
    collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(point) \
        .filterDate(start_date, end_date) \
        .sort('CLOUDY_PIXEL_PERCENTAGE')

    # Verificar se a coleção contém imagens
    if collection.size().getInfo() > 0:
        image = collection.first()
        image_date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd').getInfo()

        try:
            # Extrair valores das bandas
            bands = image.select(['B2', 'B3', 'B4', 'B5', 'B8', 'B11']).reduceRegion(
                ee.Reducer.first(), point, 10).getInfo()
            
            # Calcular índices espectrais
            ndci = (bands['B5'] - bands['B4']) / (bands['B5'] + bands['B4'])  # Índice de Clorofila Normalizado
            ndvi = (bands['B8'] - bands['B4']) / (bands['B8'] + bands['B4'])  # Índice de Vegetação por Diferença Normalizada
            fai = bands['B8'] - (bands['B4'] + (bands['B11'] - bands['B4']) * (833 - 665) / (1610 - 665))  # Índice de Algas Flutuantes
            mndwi = (bands['B3'] - bands['B11']) / (bands['B3'] + bands['B11'])  # Índice de Água por Diferença Normalizada Modificado
            b3_b2_ratio = bands['B3'] / bands['B2']  # Razão entre bandas verde e azul
            b4_b3_ratio = bands['B4'] / bands['B3']  # Razão entre bandas vermelho e verde
            b5_b4_ratio = bands['B5'] / bands['B4']  # Razão entre bandas red edge e vermelho
            
            return pd.Series({
                'B2': bands['B2'], 'B3': bands['B3'], 'B4': bands['B4'], 
                'B5': bands['B5'], 'B8': bands['B8'], 'B11': bands['B11'],
                'NDCI': ndci, 'NDVI': ndvi, 'FAI': fai, 'MNDWI': mndwi, 
                'B3_B2_ratio': b3_b2_ratio, 'B4_B3_ratio': b4_b3_ratio, 'B5_B4_ratio': b5_b4_ratio,
                'Image_Date': image_date
            })
        except Exception as e:
            print(f"Erro ao selecionar as bandas ou calcular índices para o ponto {row['PONTO']}: {e}")
            return pd.Series({col: None for col in new_columns})
    else:
        print(f"Nenhuma imagem encontrada para o ponto {row['PONTO']} ({row['Latitude']}, {row['Longitude']}) no período de {start_date.format('YYYY-MM-dd').getInfo()} a {end_date.format('YYYY-MM-dd').getInfo()}")
        return pd.Series({col: None for col in new_columns})

# Aplicar a função extract_bands_and_indices a cada linha do DataFrame
new_columns = ['B2', 'B3', 'B4', 'B5', 'B8', 'B11', 
               'NDCI', 'NDVI', 'FAI', 'MNDWI', 
               'B3_B2_ratio', 'B4_B3_ratio', 'B5_B4_ratio',
               'Image_Date']
df[new_columns] = df.apply(extract_bands_and_indices, axis=1)

# Remover linhas sem dados de imagem (valores nulos)
df_train = df.dropna(subset=new_columns, how='all')

# Adicionar colunas para mês e estação
df_train['Month'] = df_train['DATA'].dt.month
df_train['Season'] = (df_train['DATA'].dt.month % 12 + 3) // 3

# Imprimir informações sobre o dataset de treinamento
print(f"Número de amostras no dataset de treinamento: {len(df_train)}")
print(f"Colunas no dataset de treinamento: {df_train.columns.tolist()}")

# Salvar o DataFrame para treinamento
df_train.to_csv('../../sitsu_data/finished_processed_data/kinross_bandas_com_tolerancia_treino.csv', index=False)

print(f"Arquivo de treino salvo com {len(df_train)} amostras e {len(df_train.columns)} colunas.")
print(f"Colunas no arquivo de treino: {df_train.columns.tolist()}")