import torch
import os
from itertools import product
from graph_generation import *
from graph_model import *
from brute_force import *
from experiment import *
import math
import time
import ast
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

##############################
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
#############################

if not os.path.exists("./model"):
    os.makedirs("./model")
if not os.path.exists("./graph"):
    os.makedirs("./graph")
if not os.path.exists("./img"):
    os.makedirs("./img")
if not os.path.exists("./time_complexity"):
    os.makedirs("./time_complexity")


def all_experiment(params_all: dict ):
    train_num_nodes = params_all['train_num_nodes']
    test_num_nodes = params_all['test_num_nodes']
    num_epoch = params_all['num_epoch']
    device = torch.device('cuda')

    graphs, in_features = graph_generator("graph", 'random_undirected_graphs'+str(train_num_nodes) +'.pkl', 1,          10,         train_num_nodes,abs(train_num_nodes + train_num_nodes * 0.1), abs(train_num_nodes + train_num_nodes * 0.3))

    n_heads_list = [8]
    hidden_features_list = [32, 64, 128, 256]
    out_features_list = [8]
    n_epochs_list = [100]
    lr_list = [0.0005, 0.005, 0.05, 0.00005, 0.5]
    dropout_list = [0.3, 0.5, 0.7, 0.1]


    params = list(product(n_heads_list, hidden_features_list, out_features_list, n_epochs_list, lr_list, dropout_list))
    params_names = ['n_heads', 'hidden_features', 'out_features', 'n_epochs', 'lr', 'dropout']
    train_time_all = []
    for ex_num, param in enumerate(params):
        print(param)
        param_dict = {}
        for idx, param_name in enumerate(params_names):
            param_dict[param_name] = param[idx]
        gat_model, train_time, model_name = run_experiment(in_features, graphs, param_dict)
        train_time_all.append(train_time)

    return sum(train_time_all)/len(train_time_all)



train_num_nodes_list = [100]
test_num_nodes_list = [10]
num_epoch = [200]


params_all = list(product(train_num_nodes_list, test_num_nodes_list, num_epoch))
params_names = ['train_num_nodes', 'test_num_nodes', 'num_epoch']

timing_list = []
for ex_num, param in enumerate(params_all):
    print(f'Experiment {ex_num}')
    param_dict_all = {}
    for idx, param_name in enumerate(params_names):
        param_dict_all[param_name] = param[idx]
    train_time = all_experiment(param_dict_all)
    tmp = [str(param), train_time]
    timing_list.append(tmp)

training_x = []
testing_y = []
time_train_z = []
time_model_z = []
time_bf_z = []
for i in range(0, len(timing_list)):
    tuple_values = ast.literal_eval(timing_list[i][0])
    training_x.append(tuple_values[0])
    testing_y.append(tuple_values[1])
    time_train_z.append(timing_list[i][1])
print(all)
means = [sum(all[2][i] for i in range(len(all[0])) if all[0][i] == val) / all[0].count(val) if val in all[0] else None for val in train_num_nodes_list]
result_train = [train_num_nodes_list, means]
result_train[1]

plt.figure(figsize = (15,15))
plt.subplot(2,2,1)
plt.plot(train_num_nodes_list, result_train[1])
plt.xlabel("# of Training Nodes")
plt.ylabel("Time")
plt.title("Time Complexity for Training")
plt.savefig("./time_complexity/single_complexity_tr.jpg")
plt.show(block=False)