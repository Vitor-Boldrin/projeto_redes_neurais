from .camada_densa import CamadaDensa
import numpy as np
import pandas as pd
from .dados import Dados
from .log import Log
from .entropia_cruzada import EntropiaCruzada 
from .funcao_de_custo import FuncaoDeCusto

# TO DO:
# [X] função _avalia que calcula a passagem de uma amostra $h_{\theta}(x)$
# [X] função _calculo_custo computa o custo de 1 amostra
#     [X] criar a classe de custo da Regressão Logística VIROU UMA CLASSE
#     [X] avaliar a possibilidade de vetorização ESTÁ VETORIZADA
#     [X] terminar a função _calcula_custo
# [ ] implementar o backpropagation

class RedeNeural:
    def __init__(self, camadas, funcao_de_custo='entropia_cruzada', regularizador_lambda = None):
        self.regularizador_lambda = regularizador_lambda
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

        # se foi passado regularizador, passa o lambda para as camadas, pois vão ser utilizado no backpropagation
        if not(self.regularizador_lambda is None):
            for c in camadas:
                c.regularizador_lambda = self.regularizador_lambda


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
        saida = self.camadas[0]._forward(entrada)

        # Se ela só tinha uma camada, então retorna já
        if len(self.camadas) == 1:
            return saida

        for camada in self.camadas[1:]:
            saida = camada._forward(saida)

        return saida

    def _calcula_custo(self, y_real: np.array, y_pred: np.array):
        # testa se os dados são a nossa belíssima classe
        # if not isinstance(Dados_calculo, Dados):
        #    raise TypeError(f"Os dados devem ser do tipo Dados. Foi recebido: {type(Dados_calculo).__name__}")
        valor_regularizacao = 0.0

        # REGULARIZAÇÃO
        if self.regularizador_lambda is not None:
            for camada in self.camadas:
                if camada.conter_bias:
                    # Ignora a coluna 0 (BIAS) somando do indice 1 em diante
                    valor_regularizacao += np.sum(camada.parametros[:, 1:] ** 2)
                else:
                    valor_regularizacao += np.sum(camada.parametros ** 2)
            
            # Depois de somar tudo multiplica por lambda/2M 
            M = y_real.shape[1]
            valor_regularizacao = (self.regularizador_lambda / (2 * M)) * valor_regularizacao

        # Chama a função de custo de que foi definida
        self.valor_custo = self._funcao_de_custo._forward(y_real, y_pred, valor_regularizacao)

        return self.valor_custo

    def _backpropagation(self, y_real: np.array, y_pred: np.array, taxa_aprendizado: float, atualizar_pesos = True):
        """
        Faz o backpropagation uma vez e depois atualiza os thetas das camadas
        """
        # calcula os dados, passa eles pela rede
        backward = self._funcao_de_custo._backward(y_real, y_pred)

        # iterando de tras para frente
        for camada in reversed(self.camadas):
            backward = camada._backward(backward)

        # atualiza os parametros
        if atualizar_pesos:
            for camada in self.camadas:
                camada._atualiza_parametros(taxa_aprendizado)

    def _erro_gradiente_aproximado(self, X ,Y):
        eps = 1e-4
        gradientes = []
        gradientes_aprox = []

        for camada in self.camadas:
            gradientes.append(camada.d_parametros.flatten())

        gradientes_concat = np.concatenate(gradientes)
        gradientes_aprox_concat = np.zeros_like(gradientes_concat)

        idx_atual = 0

        for camada in self.camadas:
            for idx, valor in np.ndenumerate(camada.parametros):
                camada.parametros[idx] = float(valor) + eps
                y_mais = self._avalia(X)
                custo_mais = self._calcula_custo(Y, y_mais)

                camada.parametros[idx] = float(valor) - eps
                y_menos = self._avalia(X)
                custo_menos = self._calcula_custo(Y,y_menos)

                camada.parametros[idx] = float(valor)

                gradiente_calculado = (custo_mais - custo_menos) / (2 * eps)
                self.log._adicionar_log("gradiente_aproximado",gradiente_calculado)

                idx_atual += 1


    def treinar(self, epocas: int, taxa_aprendizado: float, conjunto_treinamento, conjunto_validacao = None, regularizador_lambda = None, verificacao_gradiente = False):
        """
        Junta as funções e faz o treinamento
        define as épocas que são quantas vezes será iterado
        IMPORTANTE avalia a rede antes de treinar, para atualizar todos os valores de forward na rede inteira.

        Aqui a gente pode passar o lambda da regularização também

        TO DO
        [X] GERAR LOGS DE TREINAMENTO
        """
        self.regularizador_lambda = regularizador_lambda

        # se foi passado regularizador, passa o lambda para as camadas, pois vão ser utilizado no backpropagation
        if not(self.regularizador_lambda is None):
            for c in self.camadas:
                c.regularizador_lambda = self.regularizador_lambda

        x_treino = conjunto_treinamento[0]
        y_treino = conjunto_treinamento[1]

        self.log = Log()

        if conjunto_validacao != None:
            x_validacao = conjunto_validacao[0]
            y_validacao = conjunto_validacao[1]


            for epoca in range(epocas):            
                self.log._adicionar_log("epoca",epoca)

                y_pred = self._avalia(x_treino)
                custo_treino = self._calcula_custo(y_treino, y_pred)
                self.log._adicionar_log("custo_treinamento",custo_treino)

                if verificacao_gradiente:

                    self._backpropagation(y_treino, y_pred, taxa_aprendizado, atualizar_pesos=False)

                    self._erro_gradiente_aproximado(x_treino, y_treino)

                    for camada in self.camadas:
                        camada._atualiza_parametros(taxa_aprendizado)
                else:
                    self._backpropagation(y_treino, y_pred, taxa_aprendizado)
    
                y_pred_validacao = self._avalia(x_validacao)
                custo_validacao = self._calcula_custo(y_validacao, y_pred_validacao)
                self.log._adicionar_log("custo_validacao",custo_validacao)

                self.log._salvar()
                
        else:
            for epoca in range(epocas):
                self.log._adicionar_log("epoca", epoca)      
                y_pred = self._avalia(x_treino)
                custo_treino = self._calcula_custo(y_treino, y_pred)
                self.log._adicionar_log("custo_treinamento",custo_treino)

                if verificacao_gradiente:
                    self._backpropagation(y_treino, y_pred, taxa_aprendizado, atualizar_pesos=False)

                    self._erro_gradiente_aproximado(x_treino, y_treino)

                    for camada in self.camadas:
                        camada._atualiza_parametros(taxa_aprendizado)
                else:
                    self._backpropagation(y_treino, y_pred, taxa_aprendizado)
                