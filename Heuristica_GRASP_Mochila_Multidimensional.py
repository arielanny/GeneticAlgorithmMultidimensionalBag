# OBJETIVO DO CODIGO 

# Atividade Topicos em Otimizacao Combinatoria
# Setembro 2022
# arielanny@duck.com

# Implementacao da heuristica GRASP no problema da mochila multi dimensional
# o algoritmo recebe mochilas de dimensoes nxm

# ------------------------------------------------------------------------------------------------------------
# IMPORTS
from queue import Empty
import random as rd
import numpy as np
from quicksort import quicksort, partition
from itertools import combinations

# ------------------------------------------------------------------------------------------------------------
# FUNCOES

def calcula_peso (solucao, matriz):
    # função que calcula a soma atual dos pesos para cada atributo da mochila multidimensional
    peso = []  
    
    peso = np.matmul(matriz, solucao)

    return peso     # retorna uma lista com a soma dos pesos para um atributo da mochila multidimensional

def GRASP (max_it, matriz, restricao, custos_fo, alpha=0.7):
    melhor_solucao = None
    melhor_custo = -1000000             # constante negativa qualquer, so para que pudesse comparar a primeira vez
    for i in range(max_it):
        solucao = Greedy_Randomized_Construction(matriz, restricao, custos_fo, alpha)
        # print("Custo atual:", solucao[1])
        if i == 0:
            inicial = solucao
            
        solucao, custo = Busca_Local(solucao[0], matriz, restricao, custos_fo)
        # print("Custo atual:", custo)
        
        if custo > melhor_custo:
            melhor_solucao = solucao
            melhor_custo = custo
    
    return(melhor_solucao, melhor_custo, inicial)
        
def Greedy_Randomized_Construction (matriz, capacidades, custos, alpha=0.7):
    # Heurística construtiva para a mohila multidimensional
    # Heurística pseudo-gulosa que tem variabbilidade nos melhores candidatos 
    # Usa Lista restrita de candidatos e seleciona aleatoriamente um deles
    
    solucao = np.zeros(len(custos), dtype= np.int8)
    
    n_linhas = matriz.shape[0]
    n_colunas = matriz.shape[1]
    
    continuar = True
    conti = 0              # número de iterações
    
    # enquanto eu puder ter uma lista restrita de candidatos, conseguimos rodar o algoritmo
    while continuar == True:
        conti= conti +1
            
        pesos = calcula_peso(solucao, matriz)
        # print(pesos)

        pre_selecionados = []
        
        # vendo se os elementos cabem antes de coloca-los na LRC
        evitar = []
        for i in range(n_linhas):
            for j in range(n_colunas):
                if j in evitar or solucao[i] == 1:
                    pass
                else:
                    if (matriz[i][j] + pesos[i]) <= capacidades[i]:
                        pass
                    else:
                        evitar.append(j)
        
        # tirando da lista de candidatos os que serao "evitados" e os que ja foram escolhidos
        for j in range(n_colunas):
            if j in evitar or solucao[j] == 1:
                pass
            else:
                pre_selecionados.append(j)
                
        LRC = []
        
        if len(pre_selecionados) == 0:
            continuar = False
            break
        
        # calcular o custo incremental e salvar
        selecionados = len(pre_selecionados)
        custo_incremental = []
        elems_e_custos = []    # para manter historico de onde veio o custo incremental
        for i in range(selecionados):
            solucao_aux = solucao.copy()
            solucao_aux[pre_selecionados[i]] = 1        # supondo que estamos adicionando esse elemento
            
            # print("Solução Auxiliar: ", solucao_aux)
            custo_fo = np.matmul(solucao_aux, custos)
            
            elem_custo = [pre_selecionados[i], custo_fo]
            
            custo_incremental.append(custo_fo)
            elems_e_custos.append(elem_custo)
            
            del(solucao_aux)
    
        # ordenando os custos incrementais
        candidatos_ordenados = quicksort(0, len(custo_incremental)-1, custo_incremental)
        
        # criando intervalo de construcao do LRC
        intervalo_LRC = [(candidatos_ordenados[-1] - alpha*(candidatos_ordenados[-1] - candidatos_ordenados[0])), candidatos_ordenados[-1]]
        
        # criando a LRC com indices dos elementos 
        for i in range(len(elems_e_custos)):
            if elems_e_custos[i][1] >= intervalo_LRC[0] and  elems_e_custos[i][1] <= intervalo_LRC[1]:
                LRC.append(elems_e_custos[i][0])           # lista de indices de elementos
        
        o_escolhido = rd.choice(LRC)
        
        solucao[o_escolhido] = 1
        
        del(LRC)
        
    return solucao

def verifica_factibilidade (solucao, matriz, capacidades):        
    pesos = calcula_peso(solucao, matriz)
    
    i=0
    for i in range(len(pesos)): 
        if pesos[i] <= capacidades[i]:
            continuar = True
        else:
            continuar = False
            return continuar
            
        i+=1
        
    return continuar
        
def calcula_vizinhos (solucao):
    # funcao que calcula a vizinhanca de uma solucao
    # a vizinhanca aqui foi definida como V(x) = { x != x'por dois elementos }
    
    # fazendo combinacao 2 a 2 das posicoes - para trocar o valor nelas depois
    posicoes = np.arange(0, len(solucao),1)     

    combinacoes = combinations(posicoes, 2)
    vizinhanca = []
    
    aux = solucao
    
    for i in list(combinacoes):       
        # Criando a copia para nao mudar a solucao original
        aux = solucao.copy()
        
        # Pegando posicoes que eu tenho que trocar o valor
        primeiro = i[0]
        segundo = i[1]
        
        # Troca de valor
        if solucao[primeiro] == 0:
            aux[primeiro] = 1
        else:
            aux[primeiro] = 0
        
        if solucao[segundo] == 0:
            aux[segundo] = 1
        else:
            aux[segundo] = 0

        vizinhanca.append(aux)
        
    return vizinhanca
             
def Busca_Local(sol_ini, matriz, capacidades, custos):
    pass
    solucao = sol_ini
    custo_solucao = np.matmul(sol_ini, custos)
    
    resposta = True
    while resposta == True:
        resposta = verifica_factibilidade(solucao, matriz, capacidades)

        if resposta == False:
            break
        
        # Pegando a vizinhanca da solucao atual
        vizinhanca = calcula_vizinhos(solucao)
        
        # Verificando quais dos vizinhos sao factiveis
        sol_factiveis = []
        n_vizinhos = len(vizinhanca)
        for i in range (n_vizinhos):
            fact = verifica_factibilidade(vizinhanca[i], matriz, capacidades)
            
            if fact == True:
                indice_fact = (i, vizinhanca[i])
                sol_factiveis.append(indice_fact)
                
        n_factiveis = len(sol_factiveis)
        
        if n_factiveis == 0:
            resposta = False
            print("Todos os vizinhos sao infactiveis")
            break
        
        # Comparando vizinhos factiveis com a solucao atual
        troca = 0
        for i in range (n_factiveis):
            # print("Solução factível atual", sol_factiveis[i])
            custo_fo = np.matmul(sol_factiveis[i][1], custos)
            
            if custo_fo > custo_solucao:
                solucao, custo_solucao = sol_factiveis[i][1], custo_fo
                troca += 1
        
        if troca == 0:
            resposta = False
            # print("Não há soluções factíveis melhores")
            
    return solucao
