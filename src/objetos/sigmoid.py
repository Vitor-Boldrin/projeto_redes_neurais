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
        self.valor_foward = 0.0
        self.valor_backward = 0.0

    def _foward(self, entrada: np.array):

        # também controlamos aqui
        entrada_estavel = np.clip(entrada, -400, 400)

        # calcula a softmax
        self.valor_foward = 1 / (1 + np.exp(-entrada_estavel))
        return self.valor_foward

    def _backward(self, entrada: np.array):
        """
        A derivada dela é sigmoid (1 - sigmoid) do valor calculado no foward, ou seja, o valor que ela devolveu quando rodou a rede
        """
        derivada = self.valor_foward * (1.0 - self.valor_foward)
        
        # a saida é a entrada multiplicação elemento a elemento segundo nosso amigo florindo
        self.valor_backward = entrada * derivada
        
        return self.valor_backward

