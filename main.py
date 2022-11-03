# Arquivo para testes do GA

# IMPORTS
import numpy as np
from Heuristica_GA_Mochila_Multidimensional import algoritmo_genetico, calcula_fitness
from Heuristica_GRASP_Mochila_Multidimensional import verifica_factibilidade
from instancias import *
import random as rd
import csv
import time


# LISTAS ITERAVEIS
seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]     # colocar as seeds aqui para testar (ele vai rodar automaticamente encima delas)
matrizes = [matriz_1, matriz_2, matriz_3, matriz_4, matriz_5, matriz_6, matriz_7, matriz_8, matriz_9, matriz_10]
restricoes = [restricao_1, restricao_2, restricao_3, restricao_4, restricao_5, restricao_6, restricao_7, restricao_8, restricao_9, restricao_10]
custos = [custos_fo_1, custos_fo_2, custos_fo_3, custos_fo_4, custos_fo_5, custos_fo_6, custos_fo_7, custos_fo_8, custos_fo_9, custos_fo_10]

n_matrizes = len(matrizes)

# PARAMETROS
tamanho = 100
iteracoes = 50
separacao = "sorteio"
beta= 10
alpha = 1
torneio = 3
mutacao = 0.02
porc_pais = 0.3
todas_pops = np.array(["Factibilidade", "Valor FO"])
melhor_solucao = 0


linhas = []

# LOOP DE TESTES
n_testes = 0        # Variável de controle pra ver em qual teste está
print("Total de testes a serem realizados: ", len(seeds)*len(matrizes))
tempo_total_ini = time.time()

for i in seeds:
     rd.seed(i)
     
     for k in range(n_matrizes):
          print("Realizando teste:  ", n_testes)
          
          matriz = np.asarray(matrizes[k])
          restricao = np.asarray(restricoes[k])
          custo = np.asarray(custos[k])

          inicio = time.time()
          solucao, populacao, tabela_pop = algoritmo_genetico(tamanho, iteracoes, matriz, restricao, custo,
                                            separacao, beta, alpha, torneio, mutacao, porc_pais)
          fim = time.time()
          elapsed = fim - inicio
               
          linha = [str(n_testes), str(seeds[i]), str(k+1), str(solucao), str(np.matmul(custo, solucao)), 
                        str(verifica_factibilidade(solucao, matriz, restricao)), str(elapsed),
                        str(tamanho), str(iteracoes), str(separacao), str(beta), str(alpha), str(torneio),
                        str(mutacao), str(porc_pais)]
          linhas.append(linha)

          todas_pops = np.vstack(tabela_pop)
          
          n_testes+=1
               

tempo_total_fim = time.time()
print("Tempo total:", tempo_total_fim- tempo_total_ini)

# SALVANDO OS RESULTADOS 
campos = ["# Teste", "Seed", "Instância", "Melhos solução", "Valor FO", "Factível", "Tempo", "Tam Pop", 
          "N iterações", "Mét Separação", "Beta", "Alpha", "N Torneio", "Mutação", "Pais"]

nome_arquivo = "resultados_ga.csv"

with open(nome_arquivo, 'w') as csvfile: 
    csvwriter = csv.writer(csvfile)  
    csvwriter.writerow(campos) 
    csvwriter.writerows(linhas)
    
# SALVANDO OS RESULTADOS 
campos2 = ["# Geracao", "Factibilidade", "Valor FO"]

nome_arquivo2 = "tabela_yuri_grafico.csv"

# print(todas_pops)

with open(nome_arquivo2, 'w') as csvfile: 
    csvwriter = csv.writer(csvfile,delimiter=",")  
    csvwriter.writerow(campos2) 
    csvwriter.writerows(todas_pops)
    
del(linha)
del(linhas)