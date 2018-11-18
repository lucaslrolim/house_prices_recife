# Predição de preço de imóveis na cidade de Recife

## Objetivo e contextualização

Essa análise foi desenvolvida pelo aluno Lucas Rolim como parte da disciplina de Introdução ao Aprendizado de Máquina *(EEL891)* ofertada pelo Departamento de Eletrônica da Universidade Federal do Rio de Janeiro em 2018.2.

O objetivo do trabalho é desenvolver um modelo capaz de prever o preço de imóveis na cidade de Recife. Para tal será utilizado o conjunto de dados fornecido no desafio do Kaggle da matéria e também um conjunto de dados minerado da web sobre distribuição de renda nos bairros da cidade.

Os modelos de regressão explorados na análise foram:

- Regressão polinomial
- Regressão Ridge
- Regressão Lasso
- Elastic Net
- Ridge bayessiano
- Florestas aleatórias
- LGBM
- xgBoost

## Resultados

A tabela de resultados abaixo foi calculada ao executar os modelos e calcular a média dos resultados da validação cruzada utilizando 5 folds.

Nota-se que modelos de Ensemble tiveram um desempenho significativamente melhor, de maneira geral.

**Modelo**|**RMSPE**|**desvio_padrao**
:-----:|:-----:|:-----:
Polinomial|0.255948|0.017650
Ridge|0.018744|0.000475
Lasoo|0.027953|0.000556
ElasticNet|0.027592|0.000529
BayesianRidge|0.018751|0.000484
RandomForest|0.017103|0.000692
LGBM|0.016940|0.001689
KNeighbors|0.023782|0.000740
xgBoost|0.016589|0.000486
