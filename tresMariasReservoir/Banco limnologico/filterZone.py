import pandas as pd

def remover_valores_invalidos(arquivo_entrada, arquivo_saida):
    # Carregar o CSV
    df = pd.read_csv(arquivo_entrada, dtype=str)  # Lendo os dados como string para evitar conversões inesperadas

    # Definir os valores inválidos a serem descartados
    valores_invalidos = ["NO", "ND", ".", "-","N O"]

    # Remover registros onde 'Clorofila_a_(µg_l)' está ausente, ou contém qualquer valor inválido
    df_filtrado = df.dropna(subset=["Clorofila_a_(µg_l)"])  # Remove NaN
    df_filtrado = df_filtrado[~df_filtrado["Clorofila_a_(µg_l)"].str.upper().isin(valores_invalidos)]  # Remove 'NO', 'ND', '.', '-'

    # Salvar o novo CSV
    df_filtrado.to_csv(arquivo_saida, index=False)

# Exemplo de uso
remover_valores_invalidos("BancoCodevasf_Filtrado.csv", "BancoCodevasf_Limpo.csv")
