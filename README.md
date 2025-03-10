# Water Quality AI

## Descrição
Este projeto utiliza técnicas de aprendizado de máquina para analisar e prever parâmetros de qualidade da água em diferentes reservatórios brasileiros (Caçu, São Simão e Três Marias) utilizando dados de sensoriamento remoto e medições in situ.

## Estrutura do Projeto

### Reservatório de Caçu
Contém análises de parâmetros de qualidade da água incluindo:
- Clorofila-a
- Sólidos Dissolvidos Totais
- Transparência da Água
- Turbidez

Os modelos implementados incluem:
- Random Forest
- Gradient Boosting
- SVR (Support Vector Regression)
- Lasso
- Ridge

### Reservatório de São Simão
Contém dados brutos e ferramentas de análise para o reservatório de São Simão.

### Reservatório de Três Marias
Contém dados, modelos otimizados e extrações de bandas para o reservatório de Três Marias, com foco em:
- Clorofila
- Turbidez

## Processamento de Dados

### Dados In Situ
O projeto utiliza dados in situ de diversas fontes:
- Base Kinross (Caçu)
- Relatórios Aquabase (São Simão)
- Dados da CEMIG, CODEVASF e Guaicuy (Três Marias)

### Processamento de Imagens
O projeto utiliza dados do Earth Engine para extrair bandas espectrais e gerar conjuntos de treinamento para os modelos de aprendizado de máquina:
- `band_extractor_and_training_dataset.py` - Extrai bandas e cria conjunto de treinamento
- `extract_bands.py` - Extrai bandas de imagens de satélite

## Modelos de Machine Learning

O projeto implementa diversos modelos para prever parâmetros de qualidade da água:

### Modelos Implementados
- Random Forest
- Gradient Boosting
- Support Vector Regression (SVR)
- Lasso Regression
- Ridge Regression

### Análise de Desempenho
Os resultados dos modelos são avaliados e comparados, com métricas armazenadas nos diretórios `validation_results` e `models_performance_analysis`.

## Uso do Projeto

### Pré-requisitos
Instale as dependências necessárias:
```bash
pip install -r requirements.txt
```

### Treinamento de Modelos
Para treinar modelos, utilize os scripts em cada diretório do reservatório:
```bash
python CacuReservoir/machine_learning/models/running_models.py
```

### Validação de Modelos
Para validar os modelos, use:
```bash
python CacuReservoir/machine_learning/models/validation.py
```

## Resultados e Análises

Para cada parâmetro de qualidade da água, são geradas:
- Análises de importância de características
- Comparação de desempenho entre modelos
- Métricas de validação
- Visualizações de distribuição de dados


