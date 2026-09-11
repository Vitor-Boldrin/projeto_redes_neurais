from .camada_densa import CamadaDensa

class RedeNeural:
    def __init__(self,camadas):

        # Checagem se foi passado camadas
        if not all(isinstance(c, CamadaDensa) for c in camadas):
            raise TypeError("Todas as camadas devem ser objetos da classe CamadaDensa")

        # checagem se as entradas e saída das camadas batem
        if len(camadas) != 1:
            for c in range(camadas)-1:
                if camadas[c].tamanho_saida != camadas[c+1].tamanho_entrada:
                    raise TypeError("As entradas e saidas das redes não batem")
        
        self.camadas = camadas
        