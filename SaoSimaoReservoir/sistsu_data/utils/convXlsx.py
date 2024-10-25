import pandas as pd
#Simple converter from xlsx to csv
data = pd.read_excel('../raw_data/Relatório Aquabase São Simão.xlsx')
data.to_csv('../raw_data/sao_simao.csv', index=False)


