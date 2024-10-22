import csv
from pyproj import Proj, transform

# Configurar a projeção UTM (22S ou 23S conforme necessário)
proj_utm = Proj(proj='utm', zone=22, south=True, ellps='WGS84')  # Substitua 22 pela zona correta, por exemplo, 23 se necessário
proj_wgs84 = Proj(proj='latlong', datum='WGS84')

# Abrir o arquivo original e criar um novo arquivo para os resultados
with open('../coords.csv', mode='r', encoding='utf-8', newline='') as infile, open('../coords_lat_lon.csv', mode='w', encoding='utf-8', newline='') as outfile:
    reader = csv.DictReader(infile)
    fieldnames = reader.fieldnames + ['Latitude', 'Longitude']
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)

    # Escrever o cabeçalho no novo arquivo
    writer.writeheader()

    # Converter cada linha de UTM para Latitude/Longitude
    for row in reader:
        try:
            x_utm = float(row['X'])
            y_utm = float(row['Y'])

            # Converter para lat/lon
            lon, lat = transform(proj_utm, proj_wgs84, x_utm, y_utm)

            # Adicionar as novas colunas ao dicionário de saída
            row['Latitude'] = lat
            row['Longitude'] = lon

            # Escrever a linha no novo arquivo
            writer.writerow(row)
        except Exception as e:
            print(f"Erro ao processar a linha {row}: {e}")

print('Conversão concluída. Os resultados foram salvos em coords_lat_lon.csv')
