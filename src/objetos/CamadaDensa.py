from funcoes import _sigmoid
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
    """
    def __init__(self,tamanhoEntrada:int,tamanhoSaida:int,funcaoAtivacao='sigmoid'):
        self._numero_camadas = np.random.rand(tamanhoSaida, tamanhoEntrada)

    def _foward(self,entrada:np.array):
        

    