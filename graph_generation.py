import networkx as nx
import random
import torch
from scipy import sparse as sp
import pickle
import numpy as np

def generate_random_weighted_graph(num_nodes, num_edges, in_features):
    # 방향 그래프 생성
    graph = nx.Graph()
    
    # 노드 추가
    nodes = range(num_nodes)
    graph.add_nodes_from(nodes)
    
    
    # 노드에 가중치 할당 및 노드 특징 벡터 생성
    x = torch.zeros(num_nodes, in_features)
    for node in graph.nodes:
        weight = random.uniform(0, 1)
        graph.nodes[node]['weight'] = weight
        x[node] = weight
    connected_nodes = set()
    start_node = random.choice(nodes)
    connected_nodes.add(start_node)
    for node in nodes:
        if node != start_node:
            graph.add_edge(start_node, node)
            connected_nodes.add(node)

    # 추가적인 랜덤 에지를 생성하여 num_edges를 충족
    additional_edges = num_edges - len(graph.edges)
    for _ in range(additional_edges):
        # 임의의 출발 노드와 도착 노드 선택
        source = random.choice(nodes)
        target = random.choice(nodes)
        
        if source == target or graph.has_edge(source, target):
            continue

        # 에지 추가
        graph.add_edge(source, target)

        # Generate v_prev tensor
    j = random.randint(0, num_nodes-1)

    adj_matrix = nx.adjacency_matrix(graph)
    adj_matrix_original = torch.Tensor(adj_matrix.todense())
    adj_matrix = adj_matrix + sp.eye(adj_matrix.shape[0]) # Add self-loop
    adj_tensor = torch.Tensor(adj_matrix.todense())
    adj_tensor = adj_tensor.unsqueeze(2)
    return graph, x, adj_tensor, adj_matrix_original, j



def generate_random_tree(num_nodes, in_features):
    # 무작위 트리 생성
    tree = nx.Graph()
    
    # 노드 추가
    nodes = range(num_nodes)
    tree.add_nodes_from(nodes)
    
    # 노드에 가중치 할당 및 노드 특징 벡터 생성
    x = torch.zeros(num_nodes, in_features)
    for node in tree.nodes:
        weight = random.uniform(0, 1)
        tree.nodes[node]['weight'] = weight
        x[node] = weight
    
    # 트리 구성
    edges = []
    for i in range(1, num_nodes):
        # 임의의 부모 노드 선택
        parent = random.choice(nodes[:i])
        edges.append((parent, i))
    
    tree.add_edges_from(edges)
    
    # Generate v_prev tensor
    j = random.randint(0, num_nodes - 1)

    adj_matrix = nx.adjacency_matrix(tree)
    adj_matrix_original = torch.Tensor(adj_matrix.todense())
    adj_matrix = adj_matrix + sp.eye(adj_matrix.shape[0])  # Add self-loop
    adj_matrix = adj_matrix.todense()  # CSR 행렬을 밀집 행렬로 변환
    adj_matrix = torch.Tensor(adj_matrix)  # PyTorch Tensor로 변환
    adj_tensor = adj_matrix.unsqueeze(2)
    
    return tree, x, adj_tensor, adj_matrix_original, j



def graph_generator(tree_or_graph, output_file, in_features, num_graphs, num_nodes, randint_1, randint_2):
    
    if tree_or_graph == "graph":
        graphs = []
        print("Creating Graphs... with " + str(num_nodes) + " nodes")
        for _ in range(num_graphs):
            num_nodes, num_edges = num_nodes, np.random.randint(randint_1, randint_2)
            print("Creating Graphs... with " + str(num_edges) + " edges")
            graph, x, adj_tensor, adj_matrix_original, j = generate_random_weighted_graph(num_nodes, num_edges, in_features)
            next_nodes = []
            graphs.append((x, adj_tensor, j, adj_matrix_original, next_nodes))
    
    elif tree_or_graph == "tree":
        graphs = []
        print("Creating Trees... with " + str(num_nodes) + " nodes")
        for _ in range(num_graphs):
            #num_nodes = 3000  # 트리의 노드 수를 조절할 수 있습니다.
            tree, x, adj_tensor, adj_matrix_original, j = generate_random_tree(num_nodes, in_features)
            next_nodes = []
            graphs.append((x, adj_tensor, j, adj_matrix_original, next_nodes))
    
    with open("./graph/" + output_file, 'wb') as f:
        pickle.dump(graphs, f)
    with open('./graph/' + str(output_file), 'rb') as f:
        graphs = pickle.load(f)
    
    return graphs, in_features