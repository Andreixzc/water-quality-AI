import pandas as pd
import ee
import os
import time
from datetime import datetime

"""
Este script extrai bandas espectrais e calcula índices de vegetação/água usando o Google Earth Engine.
Ele processa dados de qualidade da água com coordenadas geográficas, busca imagens de satélite correspondentes,
e calcula vários índices espectrais úteis para a análise de qualidade da água.

Funcionalidade:
1. Inicializa o Google Earth Engine.
2. Carrega o dataset mesclado com coordenadas geográficas.
3. Para cada ponto de amostra, busca a imagem de satélite mais próxima dentro de um intervalo de tempo.
4. Extrai valores de bandas espectrais e calcula índices relevantes.
5. Adiciona informações temporais (mês e estação) aos dados.
6. Adiciona a porcentagem de nuvens para cada instância.
7. Salva os dados processados em arquivo CSV para treinamento de modelos.
"""

def main():
    # Inicializar o Google Earth Engine
    print("Inicializando Google Earth Engine...")
    ee.Initialize()

    # Definir a tolerância de dias como uma variável parametrizável
    DIAS_TOLERANCIA = 30

    # Ler o arquivo CSV com os dados mesclados
    print("Carregando dataset mesclado...")
    df = pd.read_csv("merged_water_quality_data.csv")

    # Converter a coluna 'data' para o formato de data
    try:
        df['data'] = pd.to_datetime(df['data'])
        print(f"Conversão de data bem-sucedida. Formato detectado: {df['data'].iloc[0]}")
    except Exception as e:
        print(f"Erro ao converter datas: {e}")
        print("Tentando formatos alternativos...")
        try:
            # Tentar diferentes formatos de data
            df['data'] = pd.to_datetime(df['data'], format='%d/%m/%Y', errors='coerce')
            # Preencher NaTs (datas inválidas) com um método alternativo
            mask = df['data'].isna()
            df.loc[mask, 'data'] = pd.to_datetime(df.loc[mask, 'data'], format='%Y-%m-%d', errors='coerce')
            
            # Verificar se ainda há datas nulas e avisar
            if df['data'].isna().any():
                print(f"Aviso: {df['data'].isna().sum()} datas não puderam ser convertidas e serão removidas.")
                df = df.dropna(subset=['data'])
        except Exception as e2:
            print(f"Falha na tentativa alternativa: {e2}")
            print("Problemas com o formato das datas. Verificando primeiras linhas:")
            print(df['data'].head(10))
            return

    # Remover linhas com coordenadas inválidas
    print("Verificando coordenadas...")
    valid_coords = (
        (df['latitude'] >= -90) & (df['latitude'] <= 90) & 
        (df['longitude'] >= -180) & (df['longitude'] <= 180)
    )
    invalid_coords_count = (~valid_coords).sum()
    if invalid_coords_count > 0:
        print(f"Removendo {invalid_coords_count} linhas com coordenadas inválidas...")
        df = df[valid_coords]

    print(f"Dataset preparado com {len(df)} amostras.")

    def extract_bands_and_indices(row, row_index, total_rows):
        """
        Função para extrair bandas espectrais e calcular índices para um ponto específico.
        
        Args:
            row (pd.Series): Uma linha do DataFrame contendo informações de um ponto de amostra.
            row_index (int): Índice da linha atual para acompanhamento do progresso.
            total_rows (int): Total de linhas para processamento.
        
        Returns:
            pd.Series: Série com valores das bandas, índices calculados e porcentagem de nuvens,
                       ou pd.Series com valores nulos se nenhuma imagem for encontrada.
        """
        # Mostrar progresso a cada 10 registros
        if row_index % 10 == 0:
            print(f"Processando registro {row_index+1} de {total_rows} ({(row_index+1)/total_rows*100:.1f}%)...")
        
        point = ee.Geometry.Point(row['longitude'], row['latitude'])
        date = ee.Date(row['data'].strftime('%Y-%m-%d'))

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
            cloud_percentage = image.get('CLOUDY_PIXEL_PERCENTAGE').getInfo()

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
                    'Image_Date': image_date, 'Cloud_Percentage': cloud_percentage
                })
            except Exception as e:
                print(f"Erro ao processar ponto ({row['latitude']}, {row['longitude']}) - data {row['data'].strftime('%Y-%m-%d')}: {e}")
                return pd.Series({col: None for col in new_columns})
        else:
            if row_index % 50 == 0:  # Limitar mensagens de log para não sobrecarregar a saída
                print(f"Nenhuma imagem encontrada para o ponto ({row['latitude']}, {row['longitude']}) no período")
            return pd.Series({col: None for col in new_columns})

    # Definir as novas colunas que serão adicionadas
    new_columns = ['B2', 'B3', 'B4', 'B5', 'B8', 'B11', 
                'NDCI', 'NDVI', 'FAI', 'MNDWI', 
                'B3_B2_ratio', 'B4_B3_ratio', 'B5_B4_ratio',
                'Image_Date', 'Cloud_Percentage']

    print("Iniciando extração de bandas e cálculo de índices...")
    # Usar um método mais controlável para processar linha por linha
    results = []
    total_rows = len(df)
    
    for i, (idx, row) in enumerate(df.iterrows()):
        try:
            result = extract_bands_and_indices(row, i, total_rows)
            # Combinar os dados originais com os novos resultados
            combined_row = pd.concat([row, result])
            results.append(combined_row)
            
            # Salvar resultados parciais a cada 50 registros processados
            if (i+1) % 50 == 0:
                print(f"Salvando resultados parciais após {i+1} registros...")
                temp_df = pd.DataFrame(results)
                temp_df.to_csv(f'water_quality_with_bands_partial_{i+1}.csv', index=False)
            
            # Pequena pausa para evitar sobrecarga na API
            time.sleep(0.5)
        except Exception as e:
            print(f"Erro ao processar registro {i}: {e}")
            # Adicionar a linha original com valores nulos para as novas colunas
            null_results = pd.Series({col: None for col in new_columns})
            combined_row = pd.concat([row, null_results])
            results.append(combined_row)
    
    # Criar um novo DataFrame com todos os resultados
    df_result = pd.DataFrame(results)
    
    # Adicionar colunas para mês e estação baseadas na data da imagem
    print("Adicionando informações temporais...")
    
    # Converter a coluna 'Image_Date' para datetime
    df_result['Image_Date'] = pd.to_datetime(df_result['Image_Date'], errors='coerce')
    
    # Calcular mês e estação apenas onde Image_Date é válido
    mask = df_result['Image_Date'].notna()
    df_result.loc[mask, 'Month'] = df_result.loc[mask, 'Image_Date'].dt.month
    df_result.loc[mask, 'Season'] = (df_result.loc[mask, 'Image_Date'].dt.month % 12 + 3) // 3
    
    # Remover linhas sem dados de imagem (onde todos os valores de bandas são nulos)
    band_columns = ['B2', 'B3', 'B4', 'B5', 'B8', 'B11']
    df_with_images = df_result.dropna(subset=band_columns, how='all')
    
    print(f"Dataset final possui {len(df_with_images)} amostras com imagens (de um total de {len(df_result)}).")
    
    # Salvar o DataFrame completo
    print("Salvando datasets...")
    
    # Adicionar timestamp ao nome do arquivo para evitar sobrescrever
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Salvar dataset completo (mesmo aqueles sem imagens)
    df_result.to_csv(f'water_quality_with_bands_complete_{timestamp}.csv', index=False)
    
    # Salvar apenas as linhas com imagens encontradas
    df_with_images.to_csv(f'water_quality_with_bands_filtered_{timestamp}.csv', index=False)
    
    print("Processamento concluído!")
    print(f"Arquivos salvos: water_quality_with_bands_complete_{timestamp}.csv e water_quality_with_bands_filtered_{timestamp}.csv")
    
    # Imprimir estatísticas do processo
    print("\nEstatísticas:")
    print(f"Total de registros processados: {len(df)}")
    print(f"Registros com imagens encontradas: {len(df_with_images)} ({len(df_with_images)/len(df)*100:.1f}%)")
    print(f"Bandas e índices extraídos: {', '.join(new_columns)}")

if __name__ == "__main__":
    main()