import numpy as np
from .sigmoid import Sigmoid
from .funcao_ativacao import FuncaoAtivacao

# TO DO:
# [ ] Implementar o _backward para a implementação do backpropagation

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
    def __init__(self,tamanho_entrada:int,tamanho_saida:int,funcao_de_ativacao='sigmoid', conter_bias=True):
        CLASSES_DE_ATIVACAO = {
                    'entropia_cruzada': Sigmoid
                }

        self.conter_bias = conter_bias
        self.valor_foward = None
        self.valor_backward = None
        
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
            
        self.valor_foward = self.Ativacao._foward(soma) # funcao de ativacao e guarda a saida
        return self.valor_foward

    def _backward(self, entrada: np.array):
        """
        Seja Z = theta^T*x,
        L a função de custo,
        e A = g(Z) com g sendo a função de ativação

        o backward aqui já recebe delL/delA=entrada [NÃO CONFUDIR, É O GRADIENTE DO ERRO DA CAMADA SEGUINTE DELA]

        delL/delZ = delL/delA delA/delZ

        sabemos que delA/delZ = g'(Z) = self.funcao_de_ativacao._backward(entrada)

        então conseguimos calcular delL/delZ, com ele

        queremos agora (o sonhado gradiente)

        delL/delTheta = delL/delZ delZ/delTheta

        já que Z = Theta*X^T -> delZ/delTheta = X = self.entrada_epoca [é o valor que foi passado por aqui durante o _foward]

        daí 

        delL/delTheta = 1/m delL/delZ * X^T          -> precisamos dividir pelo número de amostras e fica X^T para bater certinho as dimensões

        com isso calculamos também delL/delTheta = self.d_parametros

        agora passando o erro dessa camada para a anterior [completando o trenzinho do _backward]

        delL/delX = delL/delZ delZ/delX

        novamente, como Z = Theta^T*X então delZ/delX = Theta
        """
        # 1. Pede para a ativação calcular a derivada dZ a partir de dA
        delA_delZ = self.Ativacao._backward(entrada)
        
        # Número de amostras
        m = self.entrada_epoca.shape[1]
        
        # Calcula o gradiente dos parâmetros (DeJ/DelTheta)
        self.d_parametros = (1 / m) * np.dot(delA_delZ, self.entrada_epoca.T)
        
        #Calcula o gradiente que será repassado para a camada anterior
        self.valor_backward = np.dot(self.parametros.T, delA_delZ)
        
        # não esquecer do bias
        if self.conter_bias:
            # Como a camada anterior não fez o bias, tira a primeira linha
            self.valor_backward = self.valor_backward[1:, :]
            
        return self.valor_backward

    def atualiza_parametros(self, taxa_aprendizado: float):
        """
        Depois de fazer o backpropagation damos um passo do gradiente
        """
        self.parametros = self.parametros - taxa_aprendizado * self.d_parametros
    