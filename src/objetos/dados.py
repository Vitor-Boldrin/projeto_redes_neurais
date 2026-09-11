import numpy as np
import pandas as pd
from pandas.errors import ParserError, EmptyDataError
import random

# TODO LIST
# [X] Ler aquivos .csv
# [ ] Implementar método holdout
# [ ] Implementar método K-Fold
# [ ] Implementar método Bootstrap

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

        self.pontos = [[a,b] for a,b in zip(self.pacote_dados, self.pacote_labels)]
        self.numero_pontos = len(self.pontos)

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
        

