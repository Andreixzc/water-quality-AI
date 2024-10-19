import csv

# Dicionário para armazenar as coordenadas de cada ponto de amostragem
coords_dict = {}

# Ler o arquivo coords_lat_lon e preencher o dicionário
with open('../coords_lat_lon.csv', 'r', encoding='utf-8') as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        ponto = row['Ponto de amostragem']
        latitude = row['Latitude']
        longitude = row['Longitude']
        coords_dict[ponto] = {'Latitude': latitude, 'Longitude': longitude}

# Função para processar um arquivo CSV
def process_csv(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        fieldnames = csv_reader.fieldnames + ['Latitude', 'Longitude']
        
        with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
            csv_writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            csv_writer.writeheader()
            
            for row in csv_reader:
                ponto = row['PONTO']
                if ponto in coords_dict:
                    row['Latitude'] = coords_dict[ponto]['Latitude']
                    row['Longitude'] = coords_dict[ponto]['Longitude']
                else:
                    row['Latitude'] = ''
                    row['Longitude'] = ''
                csv_writer.writerow(row)

# Processar os arquivos CSV
process_csv('../Base_kinross_all_parameters.csv', '../finished_processed_data/Base_kinross_all_parameters_updated.csv')
process_csv('../Base_kinross_filtered_parameters.csv', '../finished_processed_data/Base_kinross_filtered_parameters_updated.csv')