import objetos
import numpy as np


dados = objetos.Dados(caminho_dados, caminho_labels)
numero_pontos_totais = len(dados.conjunto_pontos)

print(f"Número de Pontos totais: {numero_pontos_totais}\n")

porcentagem_treinamento = 0.6
porcentagem_validacao = 0.2

dados._holdout(porcentagem_treinamento, porcentagem_validacao)

print("Tamanho conjunto de treinamento")
print(len(dados.conjunto_treinamento))
print("Tamanho conjunto de validação")
print(len(dados.conjunto_validacao))
print("Tamanho conjunto de testes")
print(len(dados.conjunto_teste))

print("Teste para ver se o conjunto é igual a sua parte no conjunto dos pontos embaralhados")
print(" Conjunto de treinamento")
print(f"    O conjunto tem elementos? {dados.conjunto_treinamento != []}")
if dados.conjunto_treinamento != [] :
    print(f"    É igual? {dados.conjunto_treinamento == dados.pontos_embaralhados[: int(np.ceil(numero_pontos_totais * porcentagem_treinamento))]}")

print(" Conjunto de validação")
print(f"    O conjunto tem elementos? {dados.conjunto_validacao != []}")
if dados.conjunto_validacao != [] :
    print(f"    É igual? {dados.conjunto_validacao == dados.pontos_embaralhados[int(np.ceil(numero_pontos_totais * porcentagem_treinamento)) : int(np.ceil(numero_pontos_totais * porcentagem_treinamento)) +  int(np.ceil(numero_pontos_totais * porcentagem_validacao))]}")

print(" Conjunto de teste")
print(f"    O conjunto tem elementos? {dados.conjunto_teste != []}")

if dados.conjunto_teste != [] :
    if dados.conjunto_validacao != [] and dados.conjunto_teste != []:
        print(f"    É igual? {dados.conjunto_teste == dados.pontos_embaralhados[int(np.ceil(numero_pontos_totais * porcentagem_treinamento)) +  int(np.ceil(numero_pontos_totais * porcentagem_validacao)) : ]}")
    elif dados.conjunto_teste != []:
        print(f"    É igual? {dados.conjunto_teste == dados.pontos_embaralhados[int(np.ceil(numero_pontos_totais * porcentagem_treinamento))  : ]}")
    else:
        print("ERROR")