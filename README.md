

# Random Forest

O algoritmo foi desenvolvido em `Python`, na versão `3.6.9`. Para executar, é necessário também o package `pandas`, na versão `1.1.1`.
#### Arquivo `main.py`
O arquivo `main.py` contém um exemplo da execução do algoritmo para florestas aleatórias utilizando _cross validation_. O dicionário `options` contém todos os parâmetros necessários tanto para execução do algoritmo quanto para a _KFoldValidation_. Ele pode ser executado por linha de comando simplesmente chamando `python3 main.py`.

### Outro arquivo
É possível executar também de outros arquivos, basta criar e importar as classes do algoritmo necessário, como por exemplo:
```python
from randomForest import RandomForest
import pandas as pd

options = {
    'df': pd.read_csv("benchmark.csv", sep=';'),
    'label_column': "Joga",
    'n_trees': 5,
    'bootstrap_size': 10
}
forest = RandomForest()
model = forest.train(options)

inf_data = pd.Series(["Ensolarado", "Quente", "Normal", "Verdadeiro"], index=["Tempo", "Temperatura", "Umidade", "Ventoso"], name ="InferenceData")
result = model.predict(inf_data)
```

##### Diego Dimer Rodrigues - 287690
##### Eduardo Chaves Paim - 277322
