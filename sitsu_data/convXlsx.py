import pandas as pd
#Simple converter from xlsx to csv
data = pd.read_excel('Base_kinross.xlsx')
data.to_csv('Base_kinross.csv', index=False)


