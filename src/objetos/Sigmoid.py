import numpy as np

class Sigmoid:
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

    def calcula(self,parametros:np.array,entrada:np.array):
        self.saida = 1/(1+np.e**(np.dot(parametros,entrada)))
        return self.saida