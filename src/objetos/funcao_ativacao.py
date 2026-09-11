from abc import ABC, abstractmethod
import numpy as np


class FuncaoAtivacao(ABC):
    """
    Classe mãe para todas as funções de ativação. Todas
    DEVEM herdar dessa classe forçando ter o atributo saida
    e o método _calcula
    """
    def __init__(self):
        self.saida = 0.0

    @abstractmethod
    def _calcula(self, entrada: np.array):
        pass