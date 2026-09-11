import numpy as np
from .funcao_ativacao import FuncaoAtivacao

class Sigmoid(FuncaoAtivacao):
    """
    Objeto da função sigmoid, dessa maneira conseguimos
    generalizar as funções de ativação assim como guardar
    na memória o valor dela para posteriormente utilizar
    no backpropagation
    possui o atributo: 
        saida (float) guarda o retorno da sigmoid
    possui o atributo:
        calculo (np.array, np.array)->(float)

    """

    def __init__(self):
        self.saida = 0.0

    def _calcula(self, entrada: np.array):
        # calcula a softmax
        self.saida = 1 / (1 + np.exp(-entrada))
        return self.saida