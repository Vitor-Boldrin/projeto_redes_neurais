import numpy as np

class CamadaDensa:
    """
        Objeto da camada densa de uma rede neural, seu
        objetivo é receber uma entrada, input da rede ou a 
        saida de uma outra camada, coletar esses valores
        multiplicar pelos parâmetros, utilizar uma função
        de ativação e retornar isso como a saída
        possui o atributo:
            parametros (np.array mxn)
        métodos:
            _foward (np.array mx1)->float recebe uma entrada e
            multiplica pelos parametros. Depois aplica
            a função de ativação e retorna o valor
    """
    def __init__(self,tamanho_entrada:int,tamanho_saida:int,Ativacao):
        self.parametros = np.random.rand(tamanho_saida, tamanho_entrada)
        self.Ativacao = Ativacao

    def _foward(self,entrada:np.array):
        return self.Ativacao.calcula(self.parametros,entrada)

    