import numpy as np
from .sigmoid import Sigmoid
from .funcao_ativacao import FuncaoAtivacao

# TO DO:
# [X] Implementar o _backward para a implementação do backpropagation

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
    def __init__(self,tamanho_entrada:int,tamanho_saida:int,funcao_de_ativacao='sigmoid', conter_bias=True, regularizador_lambda=None):
        CLASSES_DE_ATIVACAO = {
                    'entropia_cruzada': Sigmoid
                }

        self.conter_bias = conter_bias
        self.valor_foward = 0.0
        self.valor_backward = 0.0
        self.regularizador_lambda = regularizador_lambda
        
        # dimensões
        self.dimensao_entrada = tamanho_entrada 
        self.dimensao_saida = tamanho_saida
        self.entrada = None
        
        limite_randomico = 5

        # trata a função de ativação
        if isinstance(funcao_de_ativacao, str):
            if funcao_de_ativacao in CLASSES_DE_ATIVACAO:
                self.funcao_de_ativacao = CLASSES_DE_ATIVACAO[funcao_de_ativacao]()
            else:
                raise ValueError("Não foi passado uma função de custo válida")
        elif isinstance(funcao_de_ativacao, FuncaoAtivacao):
            # Se já passar o objeto criado vai ele memo
            self.funcao_de_ativacao = funcao_de_ativacao
        else:
            raise ValueError("Não foi passado uma função de custo válida")
        
        if self.conter_bias:
            # Se tem bias a matriz de parâmetros ganha uma coluna a mais
            # O shape passa a ser (tamanho_saida, tamanho_entrada + 1)
            self.parametros = np.random.uniform(-limite_randomico, limite_randomico, (tamanho_saida, tamanho_entrada + 1))
        else: 
            self.parametros = np.random.uniform(-limite_randomico, limite_randomico, (tamanho_saida, tamanho_entrada))

    def _foward(self,entrada:np.array):
        self.entrada_epoca = entrada
        if self.conter_bias:
            # Pega o número de colunas da entrada
            qtd_exemplos = entrada.shape[1]
            
            # Cria uma linha de "1s"
            linha_uns = np.ones((1, qtd_exemplos))
            
            # Empilha a linha de 1s em cima da entrada
            # A entrada de (400, 1) vira (401, 1) (para os dados mnist)
            entrada_com_bias = np.vstack((linha_uns, entrada))
            
            soma = np.matmul(self.parametros, entrada_com_bias) # faz a multiplicacao
        else:
            soma = np.matmul(self.parametros, entrada)
            
        self.valor_foward = self.funcao_de_ativacao._foward(soma) # funcao de ativacao e guarda a saida
        return self.valor_foward

    def _backward(self, entrada: np.array):
        """
        Fiz um pouco diferente dos slides para ficar mais modular

        Seja Z = theta^T*x,
        L a função de custo,
        e A = g(Z) com g sendo a função de ativação

        o backward aqui já recebe delJ/delA=entrada [NÃO CONFUDIR, É O GRADIENTE DO ERRO DA CAMADA SEGUINTE DELA]

        delJ/delZ = delJ/delA delA/delZ

        sabemos que delA/delZ = g'(Z) = self.funcao_de_ativacao._backward(entrada)

        então conseguimos calcular delJ/delZ, com ele

        queremos agora (o sonhado gradiente)

        delJ/delTheta = delJ/delZ delZ/delTheta

        já que Z = Theta*X^T -> delZ/delTheta = X = self.entrada_epoca [é o valor que foi passado por aqui durante o _foward]

        daí 

        delJ/delTheta = 1/m delJ/delZ * X^T          -> precisamos dividir pelo número de amostras e fica X^T para bater certinho as dimensões

        com isso calculamos também delJ/delTheta = self.d_parametros

        agora passando o erro dessa camada para a anterior [completando o trenzinho do _backward]

        delJ/delX = delJ/delZ delZ/delX

        novamente, como Z = Theta^T*X então delZ/delX = Theta
        """
        # 1. Pede para a ativação calcular a derivada dZ a partir de dA
        delA_delZ = self.funcao_de_ativacao._backward(entrada)
        
        # Número de amostras
        m = self.entrada_epoca.shape[1]
        
        # Calcula o gradiente dos parâmetros (DeJ/DelTheta)
        self.d_parametros = (1 / m) * np.dot(delA_delZ, self.entrada_epoca.T)

        # se nós temos regularizador a função fica W = W - alfa * (dW + lambda/m * W)
        if not(self.regularizador_lambda is None):
            # Cria uma matriz de zeros no exato formato de self.parametros para fazer as operações
            grad_l2 = np.zeros_like(self.parametros)
            
            if self.conter_bias: # NÃO REGULARIZAMOS O BIAS!!!!
                # Como o bias é a coluna 0 aplicamos a regularização apenas nas colunas de 1 em diante
                grad_l2[:, 1:] = (self.regularizador_lambda / m) * self.parametros[:, 1:]
            else:
                # Se não tem bias, regulariza a matriz inteira
                grad_l2 = (self.regularizador_lambda / m) * self.parametros
                
            # Soma o gradiente da regularização ao gradiente calculado anteriormente
            self.d_parametros += grad_l2
        
        #Calcula o gradiente que será repassado para a camada anterior
        self.valor_backward = np.dot(self.parametros.T, delA_delZ)
        
        # não esquecer do bias
        if self.conter_bias:
            # Como a camada anterior não fez o bias, tira a primeira linha
            self.valor_backward = self.valor_backward[1:, :]
            
        return self.valor_backward

    def _atualiza_parametros(self, taxa_aprendizado: float):
        """
        Depois de fazer o backpropagation damos um passo do gradiente. Nossos
        novos parametros estão guardados em self.d_parametros.
        """
        self.parametros = self.parametros - taxa_aprendizado * self.d_parametros
    