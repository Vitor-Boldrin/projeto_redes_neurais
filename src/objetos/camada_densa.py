import numpy as np

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
    def __init__(self,tamanho_entrada:int,tamanho_saida:int,Ativacao, conter_bias=True):
        self.conter_bias = conter_bias
        self.Ativacao = Ativacao
        self.saida = None
        
        # dimensões
        self.dimensao_entrada = tamanho_entrada 
        self.dimensao_saida = tamanho_saida
        
        limite_randomico = 5
        
        if self.conter_bias:
            # Se tem bias a matriz de parâmetros ganha uma coluna a mais
            # O shape passa a ser (tamanho_saida, tamanho_entrada + 1)
            self.parametros = np.random.uniform(-limite_randomico, limite_randomico, (tamanho_saida, tamanho_entrada + 1))
        else: 
            self.parametros = np.random.uniform(-limite_randomico, limite_randomico, (tamanho_saida, tamanho_entrada))

        
        self.dimensao_saida = tamanho_saida
        self.Ativacao = Ativacao
        self.saida = None

    def _foward(self,entrada:np.array):
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
            
        self.saida = self.Ativacao._calcula(soma) # funcao de ativacao e guarda a saida
        return self.saida
    