import numpy as np
import pandas as pd
from pandas.errors import ParserError, EmptyDataError
import random

# TODO LIST
# [X] Ler aquivos .csv
# [X] Implementar método holdout
# [ ] Implementar método K-Fold
# [ ] Implementar método Bootstrap

# RECADO DO VITOR: A rede está MUITO vetorizada, então a entrada da rede é uma matriz.
# se vamos treinar (aqui temos 10 classes) e digamos que m = 100 (numero de dados de treinamento)
# então Y = matriz 10x100 {classes}x{m}
# e X = matriz 400x100 

class Dados():
    """
    A classe Dados armazena e manipula os nossos dados.
    """

    def __init__(self, caminho_dados, caminho_rotulos):
        """
            Realiza a leitura dos nossos dados e seus respectivos rotúlos.

            Parametros:
                - caminho_dados: O caminho do arquivo .csv contendo nossos dados.
                - caminho_rotulos: O caminho do arquivo .csv contendo os rotúlos dos nossos dados.
        """

        self.pacote_dados = self._ler_dados(caminho_dados)
        self.pacote_rotulos = self._ler_dados(caminho_rotulos)

        self.conjunto_pontos = [[a,b] for a,b in zip(self.pacote_dados, self.pacote_rotulos)]
        self.numero_pontos = len(self.conjunto_pontos)

        self.conjunto_treinamento = []
        self.conjunto_validacao = []
        self.conjunto_teste = []

    def _ler_dados(self, caminho):
        """
            Método para ler um arquivo .csv

            Parametros:
                - caminho: O caminho do arquivo .csv que será lido.
            Retorna:
                - dados: Numpy array contendo as informações no arquivo .csv. 
        """

        try:
            dados = pd.read_csv(caminho, thousands=",").to_numpy()
        except FileNotFoundError:
            print(f"O Arquivo '{caminho}' não foi encontrado.")
        except EmptyDataError:
            print(f"O Arquivo '{caminho}' está vazio.")

        return dados
        
    def _holdout(self, *args):

        self.pontos_embaralhados = self.conjunto_pontos.copy()
        random.shuffle(self.pontos_embaralhados)

        if len(args) == 0:

            raise ValueError("É necessario ao menos especificar o tamanho do conjunto de treinamento.")

        elif len(args) == 1:

            if args[0] >= 1.0:
                raise ValueError("O tamanho do conjunto de treinamento não pode ultrapassar ou ser igual ao número de pontos.")

            self.conjunto_treinamento = self.pontos_embaralhados[ : int(np.ceil(self.numero_pontos*args[0]))]
            self.conjunto_validacao = []
            self.conjunto_teste = self.pontos_embaralhados[int(np.ceil(self.numero_pontos*args[0])) : ]
            
        elif len(args) == 2:

            if args[0] >= 1.0:
                raise ValueError("O tamanho do conjunto de treinamento não pode ultrapassar ou ser igual ao número de pontos.")
            elif args[1] >= 1.0:
                raise ValueError("O tamanho do conjunto de validação não pode ultrapassar ou ser igual ao número de pontos.")
            elif args[0] + args[1] >= 1.0:
                raise ValueError("A soma entre o tamanho do conjunto de treinamento e o tamanho do conjunto de validação não pode ultrapassar ou ser igual ao número de pontos.")

            self.conjunto_treinamento = self.pontos_embaralhados[ : int(np.ceil(self.numero_pontos*args[0]))]
            self.conjunto_validacao = self.pontos_embaralhados[ int(np.ceil(self.numero_pontos*args[0])) : int(np.ceil(self.numero_pontos*args[0])) + int(np.ceil(self.numero_pontos*args[1]))]
            self.conjunto_teste = self.pontos_embaralhados[int(np.ceil(self.numero_pontos*args[0])) + int(np.ceil(self.numero_pontos*args[1])) : ]
            
        else:
            raise ValueError("Muitos argumentos, era esperado o tamanho do conjunto de treinamento e o tamanho do conjunto de validação.")

