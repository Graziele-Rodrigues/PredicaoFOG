# PredicaoFOG
Modelo para detectar e prever episódios de FOG de Parkinson com base em dados de séries temporais registrados para cada paciente durante a execução de um protocolo específico, além de algumas características fornecidas do paciente.  Utilizou-se a metodologia de fatorial 2k para avaliar desempenho a depender dos fatores.

## Algoritmo 1: Análise Estatística e Visualização de Interação

#### Descrição
O algoritmo `perform_analysis_and_plot_interaction` realiza uma análise estatística de fatores e suas interações usando dados tabulares. Ele utiliza técnicas estatísticas como cálculo de soma dos quadrados, erros e efeitos dos fatores. Além disso, gera um gráfico de interação entre os dois fatores mais influentes.

#### Uso
```python
perform_analysis_and_plot_interaction(df, factor_columns, response_columns)
```

#### Dependências
- pandas
- numpy
- scipy.stats
- seaborn

#### Exemplo de Uso
```python
# Exemplo de uso
perform_analysis_and_plot_interaction(df, ['Factor1', 'Factor2'], ['Response'])
```

## Algoritmo 2: Previsão FOG

Este projeto tem como objetivo detectar episódios de "Freezing of Gait" (FOG) em dados de movimento utilizando um modelo de aprendizado de máquina. O script utiliza a biblioteca `joblib` para salvar e carregar o modelo treinado.

### Estrutura do Código

#### Parâmetros

- `WINDOW_SIZE`: Tamanho da janela para o cálculo de características.
- `SAMPLE_FRAC`: Fração dos dados a serem amostrados durante o carregamento.
- `CALC_TYPE`: Tipo de cálculo a ser realizado (não utilizado neste código).

#### Funções

1. **`load_and_preprocess_data(path, sample_frac=None)`**  
   Carrega e processa dados de arquivos CSV de um diretório ou de um único arquivo. Adiciona uma coluna que indica se houve FOG.

2. **`add_rolling_window_features(data, window_size)`**  
   Adiciona características de janela móvel para os eixos de aceleração (AccV, AccML, AccAP). Calcula kurtosis, skewness, variância e média.

3. **`run_model()`**  
   - Carrega e processa os dados de treino e teste.
   - Treina um modelo `LGBMClassifier` para detectar FOG.
   - Salva o modelo treinado em um arquivo.
   - Realiza previsões no conjunto de teste e calcula métricas de desempenho como precisão, recall e F1 Score.

#### Dependências

- `pandas`
- `joblib`
- `lightgbm`
- `sklearn`

#### Uso

1. Certifique-se de que todas as dependências estão instaladas.
2. Ajuste o caminho dos dados de treinamento e teste nas chamadas da função `load_and_preprocess_data`.
3. Execute o script para treinar o modelo e avaliar seu desempenho.

```python
run_model()
```

#### Resultados

O script exibirá a precisão, precisão, recall e F1 Score do modelo nos dados de teste.
