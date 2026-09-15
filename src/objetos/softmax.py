import numpy as np
from .funcao_ativacao import FuncaoAtivacao

class Softmax(FuncaoAtivacao):
    """
    Objeto da função softmax, dessa maneira conseguimos
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

    def _foward(self, entrada: np.array):

        exponenciais = np.exp(entrada)

        self.saida = exponenciais / np.sum(exponenciais, axis=0, keepdims=True) #faz os cálculos com as linhas
        
        return self.saida