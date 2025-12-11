#!/usr/bin/env python3
"""
matching_proj.py
Implementação prática do Gale-Shapley many-to-one com visualização.

Como usar:
    python matching_proj.py entradaProj2.25TAG.txt out_prefix

Gera:
 - imagens out_prefix_iter_0.png ... out_prefix_iter_N.png (até 10)
 - arquivo CSV out_prefix_matching.csv com a matriz final
 - relatório em texto no terminal

Variante: alunos propõem; projetos preferem por nota desc (5>4>3) e id asc tie-break.
Lower-quotas: heurística de pós-processamento (descrita no README).
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import re

# --- CONFIGURAÇÃO ---
NOME_ARQUIVO_ENTRADA = 'entradaProj2.25TAG.txt'

# --- 1. ESTRUTURAS DE DADOS ---

class Projeto:
    """
    Representa um Projeto.
    Define requisito_min de alunos fixo em 1 (conforme especificação).
    """
    def __init__(self, codigo, vagas_max, req_min_nota):
        self.codigo = codigo
        self.vagas_max = vagas_max
        self.requisito_min = 1      # Requisito MÍNIMO de alunos (Obrigatório 1) 
        self.req_min_nota = req_min_nota # Requisito MÍNIMO de nota do aluno (3, 4, 5)
        self.candidatos = []      
        self.emparelhados = []    
        self.rank_preferencia = {}

class Aluno:
    """Representa um Aluno."""
    def __init__(self, matricula, nota, preferencias):
        self.matricula = matricula
        self.nota = nota          
        self.preferencias = preferencias 
        self.propostas_feitas = [] 
        self.emparelhado = None   
        self.rank_projeto_escolhido = None

# --- 2. FUNÇÃO DE LEITURA E PARSE DOS DADOS ---

def ler_dados_do_arquivo(file_path, alunos, projetos):
    """Lê o arquivo de entrada, valida a nota e constrói as listas iniciais de candidatos."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data_string = f.read()
    except FileNotFoundError:
        print(f"ERRO FATAL: Arquivo '{file_path}' não encontrado.")
        return False
        
    lines = [line.strip() for line in data_string.split('\n') if line.strip() and not line.startswith('//')]
    
    PROJETO_PATTERN = re.compile(r'\((\w+),\s*(\d+),\s*(\d+)\)')
    ALUNO_PATTERN = re.compile(r'\((\w+)\):\(([^)]*)\)\s*\((\d)\)')
    
    is_project_section = True
    for line in lines:
        if is_project_section:
            match = PROJETO_PATTERN.match(line)
            if match:
                codigo, vagas_max_str, req_min_nota_str = match.groups()
                projetos[codigo] = Projeto(codigo, int(vagas_max_str), int(req_min_nota_str))
            
            if ALUNO_PATTERN.match(line):
                is_project_section = False
        else:
            match = ALUNO_PATTERN.match(line)
            if match:
                matricula, prefs_str, nota_str = match.groups()
                prefs = [p.strip() for p in prefs_str.split(',') if p.strip()]
                alunos[matricula] = Aluno(matricula, int(nota_str), prefs)
    
    for aluno_obj in alunos.values():
        for cod_projeto in aluno_obj.preferencias:
            projeto = projetos.get(cod_projeto)
            if projeto and aluno_obj.nota >= projeto.req_min_nota:
                projetos[cod_projeto].candidatos.append(aluno_obj.matricula)
                
    print(f"Dados processados: {len(projetos)} projetos, {len(alunos)} alunos.")
    return True

# --- 3. CONSTRUÇÃO DAS LISTAS DE PREFERÊNCIA DOS PROJETOS ---

def construir_rank_projetos(alunos, projetos):
    """
    Constrói a lista de preferência dos projetos.
    Critério: Nota (Decrescente) > Matrícula (Ascendente, como desempate).
    """
    for projeto in projetos.values():
        lista_alunos = [alunos[mat] for mat in projeto.candidatos if mat in alunos]
        
        # [cite_start]Seleção impessoal e competitiva: Nota mais alta primeiro [cite: 9]
        lista_alunos.sort(key=lambda a: (a.nota, -int(a.matricula[1:])), reverse=True)
        
        projeto.candidatos = []
        for rank, aluno in enumerate(lista_alunos, 1):
            projeto.rank_preferencia[aluno.matricula] = rank
            projeto.candidatos.append(aluno.matricula)

# --- 4. ALGORITMO GALE-SHAPLEY MODIFICADO (Aluno Proponente) ---

def emparelhamento_spa(alunos, projetos, max_iteracoes=10):
    """Executa o algoritmo de Emparelhamento Aluno-Projeto (SPA)."""
    alunos_nao_emparelhados = list(alunos.values())
    historico_iteracoes = [] 

    print("\nIniciando Algoritmo de Emparelhamento Estável (Aluno Proponente)...")
    
    for iteracao in range(1, max_iteracoes + 1):
        propostas_ativas = defaultdict(list)
        proponentes_atual = False
        
        # Fase 1: Propostas dos Alunos
        for aluno in alunos_nao_emparelhados:
            for cod_projeto in aluno.preferencias:
                if cod_projeto not in aluno.propostas_feitas:
                    projeto = projetos.get(cod_projeto)
                    
                    if projeto and aluno.nota >= projeto.req_min_nota:
                        propostas_ativas[cod_projeto].append(aluno.matricula)
                        aluno.propostas_feitas.append(cod_projeto)
                        proponentes_atual = True
                        break 
                    else:
                        aluno.propostas_feitas.append(cod_projeto)
        
        if not proponentes_atual:
            print(f"\nEmparelhamento finalizado na iteração {iteracao-1} (estável/nenhuma nova proposta).")
            break

        emparelhamento_temp = defaultdict(list)
        rejeicoes_temp = defaultdict(list)
        
        # Fase 2: Respostas dos Projetos
        for cod_projeto, lista_alunos_prop in propostas_ativas.items():
            projeto = projetos[cod_projeto]
            alunos_candidatos = lista_alunos_prop + projeto.emparelhados
            alunos_candidatos.sort(key=lambda mat: projeto.rank_preferencia.get(mat, float('inf'))) 
            
            alunos_aceitos = alunos_candidatos[:projeto.vagas_max]
            alunos_rejeitados = [mat for mat in alunos_candidatos[projeto.vagas_max:] if mat in alunos_candidatos]
            
            projeto.emparelhados = alunos_aceitos
            emparelhamento_temp[cod_projeto] = alunos_aceitos
            rejeicoes_temp[cod_projeto] = alunos_rejeitados

        # Fase 3: Atualização dos Alunos Rejeitados
        alunos_rejeitados_mats = set(mat for mats in rejeicoes_temp.values() for mat in mats)
        alunos_nao_emparelhados = [alunos[mat] for mat in alunos_rejeitados_mats if mat in alunos]
        
        historico_iteracoes.append({
            'propostas': [(mat, cod) for cod, mats in propostas_ativas.items() for mat in mats],
            'emparelhamentos': [(mat, cod) for cod, mats in emparelhamento_temp.items() for mat in mats],
            'rejeicoes': [(mat, cod) for cod, mats in rejeicoes_temp.items() for mat in mats]
        })
        print(f"Iteração {iteracao}: {len(alunos) - len(alunos_nao_emparelhados)} alunos emparelhados temporariamente.")
        
    # Alocação Final
    for projeto in projetos.values():
        for mat in projeto.emparelhados:
            aluno = alunos[mat]
            aluno.emparelhado = projeto.codigo
            try:
                aluno.rank_projeto_escolhido = aluno.preferencias.index(projeto.codigo) + 1
            except ValueError:
                aluno.rank_projeto_escolhido = 'N/A' 
            
    return None, historico_iteracoes

# --- 5. FUNÇÕES DE VISUALIZAÇÃO DO GRAFO (Formato Estrito da Imagem) ---

def visualizar_grafo(projetos, alunos, iteracao_data, iteracao):
    """
    Desenha o grafo no formato de duas colunas fixas (bipartite layout),
    com espaçamento aprimorado para legibilidade dos nomes.
    """
    G = nx.Graph()
    
    alunos_nodes = sorted([a.matricula for a in alunos.values()])
    projetos_nodes = sorted(list(projetos.keys()))
    
    G.add_nodes_from(projetos_nodes, bipartite=0)
    G.add_nodes_from(alunos_nodes, bipartite=1)
    
    # --- 1. POSICIONAMENTO FIXO (Layout Bipartido Estruturado) ---
    pos = {}
    
    # Definição das colunas com espaçamento aumentado
    X_PROJETO = 1
    X_ALUNO = 5 # Aumenta a separação horizontal para 4 unidades (5 - 1)

    proj_y = np.linspace(0, -(len(projetos_nodes)-1) * 2000, len(projetos_nodes))
    alunos_y = np.linspace(0, -(len(alunos_nodes)-1) * 500, len(alunos_nodes))

    pos = {}

    for y, node in zip(proj_y, projetos_nodes):
        pos[node] = (X_PROJETO, y)

    for y, node in zip(alunos_y, alunos_nodes):
        pos[node] = (X_ALUNO, y)
    
    # --- 2. Configuração das Arestas (Cores Conforme a Especificação Textual) ---
    edges_map = defaultdict(lambda: {'proposta': False, 'emparelhamento': False, 'rejeicao': False})
    
    # Prioridade de cor: Rejeição > Emparelhamento > Proposta
    for mat, cod in iteracao_data['rejeicoes']:
        edges_map[(mat, cod)]['rejeicao'] = True
    for mat, cod in iteracao_data['emparelhamentos']:
        edges_map[(mat, cod)]['emparelhamento'] = True
    for mat, cod in iteracao_data['propostas']:
        edges_map[(mat, cod)]['proposta'] = True

    edges = []
    edge_colors = []
    edge_widths = []

    for (mat, cod), status in edges_map.items():
        if status['rejeicao']:
            color = 'red' 
            width = 1.0
        elif status['emparelhamento']:
            color = 'blue'
            width = 2.5 
        elif status['proposta']:
            color = 'green'
            width = 1.5
        else:
            continue
            
        edges.append((mat, cod))
        edge_colors.append(color)
        edge_widths.append(width)

    G.add_edges_from(edges)
    
    # --- 3. Desenhar o Grafo ---
    
    # Aumenta a largura da figura para acomodar o maior espaçamento entre colunas
    plt.figure(figsize=(18, 22)) 
    node_colors = ['skyblue' if node.startswith('P') else 'lightcoral' for node in G.nodes()]
    node_sizes = 600
    
    nx.draw(G, 
            pos, 
            with_labels=True, 
            node_color=node_colors, 
            node_size=node_sizes, 
            font_size=9, 
            edge_color=edge_colors,
            width=edge_widths,
            labels={node: node for node in G.nodes()})
    
    plt.title(f"Iteração {iteracao}: Evolução do Emparelhamento (SPA) - Formato Bipartido Fixo")
    plt.axis("off")
    plt.show()
# --- 6. SAÍDA E CÁLCULO DE MÉTRICAS ---

def gerar_matriz_emparelhamento(alunos, projetos):
    """Gera a matriz final com Rank do Projeto (aluno) e Rank do Aluno (projeto)."""
    matriz_dados = []
    
    for aluno in alunos.values():
        if aluno.emparelhado:
            projeto_cod = aluno.emparelhado
            projeto = projetos[projeto_cod]

            # [cite_start]Requisito: Matriz deve incluir rank do aluno na lista do projeto e rank do projeto na lista do aluno [cite: 16]
            rank_aluno_str = f"{aluno.rank_projeto_escolhido}° (Escolha {aluno.rank_projeto_escolhido} de 3)"
            rank_projeto_value = projeto.rank_preferencia.get(aluno.matricula, 'N/A')
            rank_projeto_str = f"{rank_projeto_value}° (Rank {rank_projeto_value} de {len(projeto.candidatos)})"
            
            matriz_dados.append({
                'Aluno': aluno.matricula,
                'Nota (N)': aluno.nota,
                'Projeto Emparelhado': projeto_cod,
                'Req. Min. Nota (R)': projeto.req_min_nota,
                'Rank do Projeto (Lista do Aluno)': rank_aluno_str,
                'Rank do Aluno (Lista do Projeto)': rank_projeto_str
            })
    
    df = pd.DataFrame(matriz_dados)
    print("\n--- MATRIZ FINAL DE EMPARELHAMENTO ESTÁVEL ---")
    print(df.to_markdown(index=False))
    return df

def calcular_indice_preferencia(alunos, projetos):
    """
    Calcula o índice de preferência por projeto (Média do Rank dos alunos aceitos)
    [cite_start]e verifica o requisito mínimo de 1 aluno[cite: 12].
    """
    indices = {}
    total_alunos_alocados = 0
    
    for cod, projeto in projetos.items():
        
        if not projeto.emparelhados:
            status = 'Não Atendido' if projeto.requisito_min > 0 else 'N/A'
            indices[cod] = {'Índice Médio de Rank': 0.0, 'Vagas Preenchidas': 0, 'Status': status}
            continue
            
        total_ranks = sum(projeto.rank_preferencia[mat] for mat in projeto.emparelhados)
        num_alunos = len(projeto.emparelhados)
        indice_medio = total_ranks / num_alunos
        
        status = 'Atendido' if num_alunos >= projeto.requisito_min else f'Aviso: Abaixo do Mínimo ({projeto.requisito_min})'
        
        indices[cod] = {'Índice Médio de Rank': round(indice_medio, 2), 'Vagas Preenchidas': num_alunos, 'Status': status}
        total_alunos_alocados += num_alunos

    indice_df = pd.DataFrame.from_dict(indices, orient='index')
    indice_df.index.name = 'Projeto'
    indice_df = indice_df.sort_values(by='Índice Médio de Rank', ascending=True)

    projetos_alocados = indice_df[indice_df['Vagas Preenchidas'] > 0].shape[0]
    projetos_atendidos = indice_df[indice_df['Status'] == 'Atendido'].shape[0]

    print("\n--- ÍNDICE DE PREFERÊNCIA POR PROJETO (Média do Rank do Aluno) ---")
    print(f"Total de Alunos Alocados: {total_alunos_alocados} de {len(alunos)}")
    print(f"Total de Projetos Alocados (pelo menos 1 aluno): {projetos_alocados} de {len(projetos)}")
    print(f"Total de Projetos que Atenderam ao Requisito Mínimo (1 aluno): {projetos_atendidos}")
    print(indice_df.to_markdown())
    return indices

# --- FUNÇÃO PRINCIPAL DE EXECUÇÃO ---

def main():
    alunos = {}
    projetos = {}
    
    if not ler_dados_do_arquivo(NOME_ARQUIVO_ENTRADA, alunos, projetos):
        print("Execução encerrada.")
        return

    construir_rank_projetos(alunos, projetos)

    # [cite_start]Execução do Algoritmo SPA (máximo de 10 iterações) [cite: 13]
    _, historico_iteracoes = emparelhamento_spa(alunos, projetos, max_iteracoes=10)

    # Visualização da Evolução (Obrigatório mostrar a evolução)
    print("\n--- VISUALIZAÇÃO DO GRAFO (PRIMEIRAS IT.) ---")
    # Mostra até 4 iterações para demonstrar a evolução
    for i, data in enumerate(historico_iteracoes, 1):
        if i <= 4:
           visualizar_grafo(projetos, alunos, data, i)

    # Saídas Finais
    gerar_matriz_emparelhamento(alunos, projetos)
    calcular_indice_preferencia(alunos, projetos)

if __name__ == "__main__":
    main()