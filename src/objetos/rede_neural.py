from .camada_densa import CamadaDensa
import numpy as np

class RedeNeural:
    def __init__(self,camadas):

        # Checagem se foi passado camadas mesmo
        if not all(isinstance(c, CamadaDensa) for c in camadas):
            raise TypeError("Todas as camadas devem ser objetos da classe CamadaDensa")

        # checagem se as entradas e saída das camadas batem
        if len(camadas) != 1:
            for c in range(len(camadas)-1):
                if camadas[c].dimensao_saida != camadas[c+1].dimensao_entrada:
                    raise TypeError("As entradas e saidas das redes não batem")
        
        self.camadas = camadas

    def _avalia(self,entrada:np.array):
        #testa se o tamanho da entrada é compatível
        if entrada.shape[0] != self.camadas[0].dimensao_entrada:
            TypeError("As dimensões de entrada não são compatíveis com a da rede neural")

        #calcula a primeira camada
        saida = self.camadas[0]._foward(entrada)

        #Se ela só tinha uma camada, então retorna já
        if len(self.camadas) == 1:
            return saida

        for camada in self.camadas[1:]:
            saida = camada._foward(saida)
            return saida
        