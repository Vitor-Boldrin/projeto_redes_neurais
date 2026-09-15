from .camada_densa import CamadaDensa
import numpy as np
from .dados import Dados
from funcoes.custo import FuncaoDeCusto, EntropiaCruzada 

# TO DO:
# [X] função _avalia que calcula a passagem de uma amostra $h_{\theta}(x)$
# [ ] função _calculo_custo computa o custo de 1 amostra
#     [X] criar a classe de custo da Regressão Logística VIROU UMA CLASSE
#     [X] avaliar a possibilidade de vetorização ESTÁ VETORIZADA
#     [X] terminar a função _calcula_custo
# [ ] implementar o backpropagation

class RedeNeural:
    def __init__(self, camadas, funcao_de_custo='entropia_cruzada'):
        # dicionário de funcao de custo (para quando passar uma string)
        CLASSES_DE_CUSTO = {
            'entropia_cruzada': EntropiaCruzada
        }

        # Checagem se foi passado camadas mesmo
        if not all(isinstance(c, CamadaDensa) for c in camadas):
            raise TypeError("Todas as camadas devem ser objetos da classe CamadaDensa")

        # checagem se as entradas e saída das camadas batem
        if len(camadas) != 1:
            for c in range(len(camadas)-1):
                if camadas[c].dimensao_saida != camadas[c+1].dimensao_entrada:
                    raise TypeError("As entradas e saidas das redes não batem")

        # trata a função de custo
        if isinstance(funcao_de_custo, str):
            if funcao_de_custo in CLASSES_DE_CUSTO:
                self._funcao_de_custo = CLASSES_DE_CUSTO[funcao_de_custo]()
            else:
                raise ValueError("Não foi passado uma função de custo válida")
        elif isinstance(funcao_de_custo, FuncaoDeCusto):
            # Se já passar o objeto criado vai ele memo
            self._funcao_de_custo = funcao_de_custo
        else:
            raise ValueError("Não foi passado uma função de custo válida")

        self.valor_custo = None
        self.camadas = camadas

    def _avalia(self, entrada: np.array):
        # testa se o tamanho da entrada é compatível
        if entrada.shape[0] != self.camadas[0].dimensao_entrada:
            raise TypeError("As dimensões de entrada não são compatíveis com a da rede neural")

        # calcula a primeira camada
        saida = self.camadas[0]._foward(entrada)

        # Se ela só tinha uma camada, então retorna já
        if len(self.camadas) == 1:
            return saida

        for camada in self.camadas[1:]:
            saida = camada._foward(saida)

        return saida

    def _calcula_custo(self, y_real: np.array, y_pred: np.array):
        # testa se os dados são a nossa belíssima classe
        # if not isinstance(Dados_calculo, Dados):
        #    raise TypeError(f"Os dados devem ser do tipo Dados. Foi recebido: {type(Dados_calculo).__name__}")

        # Chama a função de custo de que foi definida
        self.valor_custo = self._funcao_de_custo.forward(y_real, y_pred)

        return self.valor_custo