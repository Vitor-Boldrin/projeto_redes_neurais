import objetos
import numpy as np

dados = objetos.Dados(caminho_dados, caminho_labels)

print(dados._separacao_holdout())