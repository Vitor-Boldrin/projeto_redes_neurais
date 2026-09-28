import pandas as pd

class Log():
    def __init__(self):
        self.data = pd.DataFrame()
        self.colunas = []
        self.data_dict = {}
        self.data.to_csv("log_treinamento.csv", index=False, header=None)

    def _reset(self):
        self.data = pd.DataFrame()
        self.data_dict = {}

    def _adicionar_log(self, coluna: str, valor):
        if(coluna in self.colunas):
            self.data_dict[f"{coluna}"] = [valor]
        else:
            self.colunas.append(coluna)
            self.data_dict[f"{coluna}"] = [valor]
            
            self.data = pd.DataFrame(columns=self.colunas)
            self.data.to_csv("log_treinamento.csv", index=False)

    def _salvar(self):
        self.data = pd.DataFrame(self.data_dict)
        self.data.to_csv("log_treinamento.csv", mode="a", index=False, header=None)
        self._reset()