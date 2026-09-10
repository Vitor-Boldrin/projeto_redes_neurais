class RedeNeural:
    def __init__(self,numero_camadas):
        self._numero_camadas = numero_camadas
        # atributos

    @property
    def numero_camadas(self):
        return self._numero_camadas

    @numero_camadas.setter
    def numero_camadas(self, valor):
        print('Você não pode alterar esse valor')

    # exemplo método
    def somar_dez_camadas(self) -> int:
        self.numero_camadas += 10
        print(f"Novo número de camadas: {self.numero_camadas}")
        return self.numero_camadas