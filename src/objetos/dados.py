import numpy as np
import pandas as pd
from pandas.errors import ParserError, EmptyDataError

# TODO LIST
# [X] Ler aquivos .csv
# [X] Implementar método holdout
# [ ] Implementar método K-Fold
# [ ] Implementar método Bootstrap

# RECADO DO VITOR: A rede está MUITO vetorizada, então a entrada da rede é uma matriz.
# se vamos treinar (aqui temos 10 classes) e digamos que m = 100 (numero de dados de treinamento)
# então Y = matriz 10x100 {classes}x{m}
# e X = matriz 400x100 

# Feito - Marcelo J.

class Dados():
    """
    A classe Dados armazena e manipula os nossos dados.
    """

    def __init__(self, caminho_dados, caminho_rotulos, numero_classes = 10):
        """
            Realiza a leitura dos nossos dados e seus respectivos rotúlos.

            Parametros:
                - caminho_dados: O caminho do arquivo .csv contendo nossos dados.
                - caminho_rotulos: O caminho do arquivo .csv contendo os rotúlos dos nossos dados.
        """

        self.pacote_dados = self._ler_dados(caminho_dados)
        self.pacote_rotulos = self._ler_dados(caminho_rotulos)
        self.numero_classes = numero_classes
        self.rotulos_originais = self.pacote_rotulos.reshape(-1).astype(int)

        self.one_hot_rotulos = np.eye(numero_classes)[self.rotulos_originais - 1]

        self.conjunto_treinamento = None
        self.conjunto_validacao = None
        self.conjunto_teste = None

    def _ler_dados(self, caminho):
        """
            Método para ler um arquivo .csv

            Parametros:
                - caminho: O caminho do arquivo .csv que será lido.
            Retorna:
                - dados: dataframe contendo as informações no arquivo .csv. 
        """

        try:
            dados = pd.read_csv(caminho, decimal=",", header=None).to_numpy()
        except FileNotFoundError:
            print(f"O Arquivo '{caminho}' não foi encontrado.")
        except EmptyDataError:
            print(f"O Arquivo '{caminho}' está vazio.")

        return dados

    def _holdout(self, *args):

        self.conjunto_treinamento = None
        self.conjunto_validacao = None
        self.conjunto_teste = None

        # Embaralha os dados e os rotulos, 
        permutacao = np.random.permutation(self.pacote_dados.shape[0])
        self.dados_embaralhados = self.pacote_dados[permutacao]
        self.one_hot_rotulos_embaralhados = self.one_hot_rotulos[permutacao]

        if len(args) == 0:

            raise ValueError("É necessario ao menos especificar o tamanho do conjunto de treinamento.")

        elif len(args) == 1:

            if args[0] >= 1.0:
                raise ValueError("O tamanho do conjunto de treinamento não pode ultrapassar ou ser igual ao número de pontos.")

            dados_treinamento, dados_teste = np.split(self.dados_embaralhados,[ int(self.dados_embaralhados.shape[0] * args[0])])
            one_hot_rotulos_treinamento, one_hot_rotulos_teste = np.split(self.one_hot_rotulos_embaralhados, [int(self.one_hot_rotulos_embaralhados.shape[0] * args[0])])

            self.conjunto_treinamento = (dados_treinamento.T, one_hot_rotulos_treinamento.T)
            self.conjunto_teste = (dados_teste.T, one_hot_rotulos_teste.T)

        elif len(args) == 2:

            if args[0] >= 1.0:
                raise ValueError("O tamanho do conjunto de treinamento não pode ultrapassar ou ser igual ao número de pontos.")
            elif args[1] >= 1.0:
                raise ValueError("O tamanho do conjunto de validação não pode ultrapassar ou ser igual ao número de pontos.")
            elif args[0] + args[1] >= 1.0:
                raise ValueError("A soma entre o tamanho do conjunto de treinamento e o tamanho do conjunto de validação não pode ultrapassar ou ser igual ao número de pontos.")

            corte1 = int(np.round(self.dados_embaralhados.shape[0] * args[0]))
            corte2 = int(np.round(self.dados_embaralhados.shape[0] * (args[0] + args[1])))

            #dados_treinamento, resto = np.split(self.dados_embaralhados, [int(self.dados_embaralhados.shape[0] * args[0])])
            #dados_validacao, dados_teste = np.split(resto, [int(resto.shape[0] * args[1])])

            dados_treinamento, dados_validacao, dados_teste = np.split(self.dados_embaralhados, [corte1, corte2])

            #one_hot_rotulos_treinamento, resto = np.split(self.one_hot_rotulos_embaralhados, [int(self.one_hot_rotulos_embaralhados.shape[0] * args[0])])
            #one_hot_rotulos_validacao, one_hot_rotulos_teste = np.split(resto, [int(resto.shape[0] * args[1])])

            one_hot_rotulos_treinamento, one_hot_rotulos_validacao, one_hot_rotulos_teste  = np.split(self.one_hot_rotulos_embaralhados, [corte1, corte2])

            self.conjunto_treinamento = (dados_treinamento.T,one_hot_rotulos_treinamento.T)
            self.conjunto_validacao = (dados_validacao.T,one_hot_rotulos_validacao.T)
            self.conjunto_teste = (dados_teste.T, one_hot_rotulos_teste.T)
            
            
        else:
            raise ValueError("Muitos argumentos, era esperado o tamanho do conjunto de treinamento e o tamanho do conjunto de validação.")

