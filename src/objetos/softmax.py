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
        self.valor_foward = 0.0
        self.valor_backward = 0.0

    def _foward(self, entrada: np.array):

        # python ta reclamando muito e parece que isso aqui ajuda
        #subtrai o maior valor do vetor para cada entrada
        entrada_estavel = entrada - np.max(entrada, axis=0, keepdims=True)

        exponenciais = np.exp(entrada_estavel)

        self.valor_foward = exponenciais / np.sum(exponenciais, axis=0, keepdims=True) #faz os cálculos com as linhas
        
        return self.valor_foward

    def _backward(self, entrada: np.array):
        """
        Calcula a derivada da função Softmax propagando o gradiente dA
        dA: Gradiente da camada posterior
        """
        A_dA = self.valor_foward * entrada

        soma_A_dA = np.sum(A_dA, axis=0, keepdims=True)
        
        self.valor_backward = A_dA - self.valor_foward * soma_A_dA
        
        return self.valor_backward