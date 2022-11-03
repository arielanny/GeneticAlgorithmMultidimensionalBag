# OBJETIVO DO CODIGO 

# Trabalho 1 Topicos em Otimizacao Combinatoria
# Outubro/Novembro 2022
# arielanny@duck.com

# Heurística Algoritmo Genético no problema da mochila multi dimensional
# o algoritmo recebe mochilas de dimensoes nxm

# ----------------------------------------------------------------------------------------------

# IMPORTS
import numpy as np
import random as rd
import pandas as pd
from Heuristica_GRASP_Mochila_Multidimensional import Greedy_Randomized_Construction
from Heuristica_GRASP_Mochila_Multidimensional import calcula_peso
from Heuristica_GRASP_Mochila_Multidimensional import verifica_factibilidade

# ----------------------------------------------------------------------------------------------

# FUNÇÕES

def gerar_população(n, A, b, c, alpha):
    #  Função que gera invidíduos aleatórios factíveis usando a heurítica construtora Greedy_Randomized_Construction
    
    # n: tamanho da população
    # A: matriz do modelo
    # b: capacidades (restrições)
    # c: vetor de custos 
    # alpha: parâmetro de aleatoriedade do pseudo guloso
    
    populacao = [] 

    # gerando a população usando a construção pseudo gulosa 
        # sempre temos uma população inicial factível
    for k in range(n):
        a = Greedy_Randomized_Construction(A, b, c, alpha)
        populacao.append(a)

    populacao = np.asarray(populacao)

    return populacao



def calcula_fitness(indv, A, b, c, beta):
    # Função fitness depende da função objetivo e da infactibilidade dos individuos
        # Um individuo infactivel leva penalidade de 0.5 subtraído de seu valor de função
        # Quanto mais infactível, pior será sua performance, e menor sua chance de ser escolhido para próxima população
    
    # indv: individuo a ser avaliado
    # A: matriz do modelo
    # b: capacidades (restrições)
    # c: vetor de custos 
    # beta: parâmetro de penalidade para a infactibilidade de um individuo
    
    valor_fo = np.matmul(c, indv)
    pesos = calcula_peso(indv, A)
    
    # calculando a infactibilidade de um individuo (calculando quanto ele excede da capacidade da mochila)
    excessos = 0
    for i in range(A.shape[0]):
        diferenca = b[i] - pesos[i]
        if diferenca < 0:
            excessos += diferenca
        
    fitness = valor_fo + beta*excessos

    return fitness



def rank_populacao(pop, A, b, c, beta):
    # Função que calcula fitness da populaçao e ordena crescentemente
    
    # pop: população de soluções
    # A: matriz do modelo
    # b: capacidades (restrições)
    # c: vetor de custos 
    # beta: parâmetro de penalidade para a infactibilidade de um individuo
    
    dic = {}                # estou usando um dicionario para manter a conexao entre a posicao e fitness
    n_populcao = len(pop)
    
    for i in range(n_populcao):
        dic[i] = calcula_fitness(pop[i], A, b, c, beta)
    
    # ordenando os indivíduos de acordo com seu fitness
    rank = sorted(dic.items(), key=lambda x:x[1])
    
    return rank



def seleciona_pais(pop, A, b, c, beta, r, ptg):
    # Função que seleciona os pais para a reprodução da próxima geração
        # Estrategia: escolher 3 aleatoriamente e pegar o melhor deles
    
    # pop: população de soluções
    # A: matriz do modelo
    # b: capacidades (restrições)
    # c: vetor de custos 
    # beta: parâmetro de penalidade para a infactibilidade de um individuo
    # r: número de individuos selecionados para torneio
    # ptg: porcentagem de pais da população escolhidos

    candidatos = list(range(0, len(pop)))     # lista de individuos disponiveis para reproducao
    reprodutores = []                               # lista de indiviuos escolhidos para reproducao
    finalizar = False
    n_pais = int(ptg*len(pop))               # estou escolhendo metade dos individuos para reproduzir

    if n_pais%2 != 0:                               # escolhendo um numero par de pais 
        n_pais +=1
    iter = 0                            
    
    # até que acabe a cota de pais, vamos sortear 3 e escolher o melhor
    while finalizar == False:
        melhor = -100000000                                # armazenar o melhor individuo
        custo = -100000000                                 # armazenar a pontuacao do melhor individuo
        escolhidos = rd.sample(candidatos, r)       # escolhendo 3 candidatos sem reposicao
            
        for a in escolhidos:
            aux = calcula_fitness(pop[a], A, b, c, beta)

            if aux > custo:
                melhor, custo = a, aux
        
        # adicionando o individuo escolhido e retirando ele como candidato
        if (melhor in reprodutores) or (melhor in candidatos):
            reprodutores.append(melhor)    
            candidatos.remove(melhor)
        
        iter = iter+1
        if iter >= n_pais:
            finalizar = True
    
    return reprodutores



def crossover(pop, A, b, c, beta, sep, r, ptg):
    # Função para fazer o cruzamento entre dois individuos
        # modalidade "metade": divide e combina metades dos pais
        # modalidade "tercos": divide e combina os terços dos pais
        # modalidade "sorteio": sorteia-se uma posição para dividir o cromossomo e combinamos as duas partes posteriormente
    
    # pop: população de soluções
    # A: matriz do modelo
    # b: capacidades (restrições)
    # c: vetor de custos 
    # beta: parâmetro de penalidade para a infactibilidade de um individuo
    # sep: método de separação dos pais
    # r: número de individuos selecionados para torneio
    # ptg: porcentagem de pais da população escolhidos
    
    
    reprodutores = seleciona_pais(pop, A, b, c, beta, r, ptg)

    pares_pais = []
    filhos = []
    
    # Divisão dos pais    
    while len(reprodutores) > 0:
        mae, pai = rd.sample(reprodutores, 2)
        pares_pais.append((mae, pai))

        reprodutores.remove(mae)
        reprodutores.remove(pai)

    n_pais = len(pares_pais)

    if sep == "metade":
        # para todos os pares de pais vamos dividir o array no meio e combinar as partes
        for i in range(n_pais):
            ind1, ind2 = pares_pais[i]

            mae = pop[ind1]
            pai = pop[ind2]
            
            parte_1_mae, parte_2_mae = np.array_split(mae,2)
            
            parte_1_pai, parte_2_pai = np.array_split(pai,2)
            
            filho_1 = np.hstack((parte_1_mae, parte_2_pai))
            filho_2 = np.hstack((parte_1_pai, parte_2_mae))

            filhos.append(filho_1)
            filhos.append(filho_2)
            
            
    elif sep == "tercos":
        # para todos os pares de pais vamos dividir o array em tres e combinar as partes
        for i in range(n_pais):
            ind1, ind2 = pares_pais[i]
            mae = pop[ind1]
            pai = pop[ind2]
            
            parte_1_mae, parte_2_mae, parte_3_mae = np.array_split(mae,3)
            
            parte_1_pai, parte_2_pai, parte_3_pai = np.array_split(pai,3)
            
            filho_1 = np.hstack((parte_1_mae, parte_2_pai, parte_3_mae))
            filho_2 = np.hstack((parte_1_pai, parte_2_mae, parte_3_pai))
            
            filhos.append(filho_1)
            filhos.append(filho_2)
            
    elif sep == "sorteio":
        # para todos os pares de pais vamos dividir o array dependendo do sorteio de posicoes
        
        for i in range(n_pais):
            posicao = rd.randint(0, len(pop)-1)

            ind1, ind2 = pares_pais[i]
            mae = pop[ind1]
            pai = pop[ind2]

            parte_1_mae, parte_2_mae = mae[0:posicao], mae[posicao:len(mae)]
            
            parte_1_pai, parte_2_pai = pai[0:posicao], pai[posicao:len(pai)]
            
            filho_1 = np.hstack((parte_1_mae, parte_2_pai))
            filho_2 = np.hstack((parte_1_pai, parte_2_mae))

            filhos.append(filho_1)
            filhos.append(filho_2)
    
    nova_pop = np.vstack((pop, filhos))           # https://pythonguides.com/python-concatenate-arrays/
    
    return nova_pop



def selecao_natural(n, pop, A, b, c, beta):
    # Função que realiza  seleção natural dos indivíduos baseado no seu fitness
        # Estratégia: os piores indivíduos são substituídos pelos filhos dos pais da geração anterior
    
    # n: tamanho da população
    # pop: população de soluções
    # A: matriz do modelo
    # b: capacidades (restrições)
    # c: vetor de custos 
    # beta: parâmetro de penalidade para a infactibilidade de um individuo


    rank = rank_populacao(pop, A, b, c, beta)
    
    excedentes = len(rank) - n
    
    # ordenando e removendo 
    pop = pop.tolist()
    for i in range(excedentes):
        rank = rank_populacao(pop, A, b, c, beta)
        del pop[rank[0][0]]
    
    pop = np.asarray(pop)
    
    return pop



def mutacao(indv):
    # Função que escolhe aleatoriamente uma posicao do dna no individuo para mudar
    
    # indv: solução
    
    genes = list(range(0, len(indv)-1))
    
    mudar = rd.choice(genes)
    
    if indv[mudar]== 0:
        indv[mudar] = 1
    else:
        indv[mudar] = 0
    
    return



def algoritmo_genetico (n, it_max, A, b, c, sep, beta, alpha, r, mut, ptg):
    # n: tamanho da população
    # it_max: número máximo de iterações
    # A: matriz do modelo
    # b: capacidades (restrições)
    # c: vetor de custos 
    # sep: método de separação dos pais
    # beta: parâmetro de penalidade para a infactibilidade de um individuo
    # alpha: parâmetro de aleatoriedade do pseudo guloso
    # r: número de individuos selecionados para torneio
    # mut: porcentagem de mutações na população
    # ptg: porcentagem de pais da população escolhidos
    
    populacao = gerar_população(n, A, b, c, alpha)
    fim = False
    ite = 0
    melhor_solucao = 0
    melhor_valor = 0
    tabela_valores = []
    
    while fim == False:
        populacao = crossover(populacao, A, b, c, beta, sep, r, ptg) 
        
        excedentes = len(populacao) - n
        
        # Aplicando mutação nos filhos (quando sorteados)
        i = excedentes-1
        while i >= 0:
            chance = rd.random()
            if chance <= mut:
                mutacao(populacao[i])
            i-=1
        
        populacao = selecao_natural(n, populacao, A, b, c, beta)  # primerio muta depois seleciona
        
        final_rank = rank_populacao(populacao, A, b, c, beta)

        melhor, valor = populacao[final_rank[-1][0]], final_rank[-1][1]
        
        for i in populacao:
            aux1 = verifica_factibilidade(i, A, b)
            aux2 = calcula_fitness(i, A, b, c, beta)
            linha = [aux1, aux2]
            tabela_valores.append(linha)
        
        ite+=1 
        
        if valor > melhor_valor:
            melhor_solucao = melhor
            melhor_valor = valor

        if ite >= it_max:
            fim = True
            
        
    
    return melhor_solucao, populacao, tabela_valores
