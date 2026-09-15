import numpy as np

def categorical_cross_entropy(y_real:np.array, y_pred:np.array):
    """
    Função da entropia cruzada para o cálculo de n classes.
    IMPORTATE: Essa função já espera estar recebendo uma matriz dos y reais e calculado.
    Ex: No nosso caso a matriz de entrada dos 2 y's é 10x{numero_de_amostras_calculadas} em que
        cada coluna é um dado de treinamento e as linhas são as classes.
    """

    # da um clip (limita) os valores para log(0) não explodir o note de voces
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    # numero de amostras pra calcular a média
    N = y_real.shape[1]
    
    # calcula
    custo = - (1 / N) * np.sum(y_real * np.log(y_pred))
    return custo