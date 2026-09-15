from .camada_densa import CamadaDensa
import numpy as np
from .dados import Dados
from funcoes.custo import entropia_cruzada

# TO DO:
# [X] função _avalia que calcula a passagem de uma amostra $h_{\theta}(x)$
# [ ] função _calculo_custo computa o custo de 1 amostra
#     [ ] criar a classe de custo da Regressão Logística
#     [ ] avaliar a possibilidade de vetorização
#     [ ] terminar a função _calcula_custo
# [ ] função _calculo_perda que calcula o custo de todos os dados de treinamento e tira a media
# [ ] implementar o backpropagation

class RedeNeural:
    def __init__(self,camadas,funcao_de_custo='entropia_cruzada'):
        FUNCOES_DE_CUSTO = {'entropia_cruzada': entropia_cruzada }

        # Checagem se foi passado camadas mesmo
        if not all(isinstance(c, CamadaDensa) for c in camadas):
            raise TypeError("Todas as camadas devem ser objetos da classe CamadaDensa")

        # checagem se as entradas e saída das camadas batem
        if len(camadas) != 1:
            for c in range(len(camadas)-1):
                if camadas[c].dimensao_saida != camadas[c+1].dimensao_entrada:
                    raise TypeError("As entradas e saidas das redes não batem")

        if funcao_de_custo in FUNCOES_DE_CUSTO:
            self._funcao_de_custo = entropia_cruzada
        else:
            raise ValueError("Função de custo inválida")

        self.valor_custo = None
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

    
    def _calcula_custo(self,y_real:np.array, y_pred:np.array):
        # testa se os dados são a nossa bélissima classa
        #if not isinstance(Dados_calculo, Dados):
        #    raise TypeError(f"Os dados devem ser do tipo Dados. Foi recebido: {type(Dados_calculo).__name__}")

        self.valor_custo = entropia_cruzada(y_real,y_pred)

        return self.valor_custo

    

    