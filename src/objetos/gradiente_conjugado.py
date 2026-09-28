import numpy as np
from .rede_neural import RedeNeural
import pandas as pd
from scipy.optimize import minimize

class GradienteConjugado:

    def __init__(self, rede_neural: RedeNeural, epocas: int, lambda_reg : float):
        self.rede = rede_neural
        self.epocas = epocas
        self.lambda_reg = lambda_reg
        self.iteração = 0

    def _vetorizar_pesos(self) -> np.ndarray:
        #Realiza a vetorização dos pesos de todas as camadas em um unico vetor
        pesos = [camada.parametros.flatten() for camada in self.rede.camadas]
        return np.concatenate(pesos)

    def _vetorizar_gradiente(self) -> np.ndarray:
        #Realiza a vetorização do gradiente de todas as camadas em um unico vetor
        gradientes = [camada.d_parametros.flatten() for camada in self.rede.camadas]
        return np.concatenate(gradientes)

    def _desvetorizar(self, vetor_1d: np.ndarray):
        #Recorta o vetor unidimensional e insere nas matrizes da rede
        indice = 0

        for camada in self.rede.camadas:
            tamanho_matriz = camada.parametros.size
            formato_matriz = camada.parametros.shape

            #realiza o recorte de acordo com o tamanho da matriz
            recorte = vetor_1d[indice:indice + tamanho_matriz]
            camada.parametros[:] = recorte.reshape(formato_matriz).copy()

            indice += tamanho_matriz

    def _função_objetivo(self, vet_pesos: np.ndarray, entrada: np.array, y_real: np.array):
        self._desvetorizar(vet_pesos)

        #Forward propagation
        y_prev = self.rede._avalia(entrada)
        print(y_prev.shape)
        custo = self.rede._calcula_custo(y_real, y_prev)

        #Backpropagation
        self.rede._backpropagation(y_real, y_prev, taxa_aprendizado=0.0, atualizar_pesos=False)

        gradiente_1d = self._vetorizar_gradiente()

        return custo, gradiente_1d

    def _print_it(self, vet_pesos: np.ndarray):
        self.iteração += 1

        if self.iteração % 10 == 0:
            print(f"Iteração {self.iteração}")

    def treino(self, entrada: np.array, y_real: np.array):
        pesos_iniciais = self._vetorizar_pesos()
        self.iteração = 0

        resultado = minimize(
            fun=self._função_objetivo,
            x0=pesos_iniciais,
            args=(entrada, y_real),
            method='CG',
            jac=True,
            callback=self._print_it,
            options={'maxiter': self.epocas}
        )

        self._desvetorizar(resultado.x)
        print(f"Treinamento concluído. Sucesso: {resultado.success}. Motivo: {resultado.message}")

        y_previsto_final = self.rede._avalia(entrada)
        print("\n================================")
        print(y_previsto_final)
        print("================================\n")

        print(resultado.fun)

