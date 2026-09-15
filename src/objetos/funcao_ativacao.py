from abc import ABC, abstractmethod
import numpy as np


class FuncaoAtivacao(ABC):
    """
    Classe mãe para todas as funções de ativação. Todas
    DEVEM herdar dessa classe forçando ter o atributo saida
    e os métodos _foward e _backward
    """
    def __init__(self):
        self.valor_foward = 0.0
        self.valor_backward = 0.0

    @abstractmethod
    def _foward(self, entrada: np.array):
        pass

    @abstractmethod
    def _backward(self, entrada: np.array):
        pass