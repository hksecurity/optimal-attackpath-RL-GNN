from graph_model import *
from tqdm import tqdm
import torch.nn.utils as torch_utils
import random
import matplotlib.pyplot as plt
import time
import math

def run_experiment(in_features, graphs, params: dict):
    n_heads = params['n_heads']
    hidden_features = params['hidden_features']
    out_features = params['out_features']
    n_epochs = params['n_epochs']
    lr = params['lr']
    dropout = params['dropout']

    d_h = 16 * n_heads


    gat_model = GAT(in_features, hidden_features, out_features, n_heads, d_h, dropout).cuda()

    optimizer = torch.optim.Adam(gat_model.parameters(), lr=lr)

    # baseline model
    baseline_model = GAT(in_features, hidden_features, out_features, n_heads, d_h, dropout).cuda()
    baseline_update_period = 40

    # fig, axes = plt.subplots(len(graphs), 1, figsize=(8, 6*len(graphs)))
    # fig1, axes1 = plt.subplots(len(graphs), 1, figsize=(8, 6*len(graphs)))
    reward_values = [[] for _ in range(len(graphs))]
    loss_lists = [[] for _ in range(len(graphs))]
    loss_mean_values = []
    reward_mean_values = []
    start_train_time = time.time()
    for epoch in tqdm(range(n_epochs)):
        for graph_idx, (x, adj_tensor, j, adj_matrix_original, _) in enumerate(graphs):
            x = x.cuda()
            adj_tensor = adj_tensor.cuda()
            loss_list =[]
            num_nodes = x.size(0)

            # epoch_segment = min(epoch // (n_epochs // num_nodes), num_nodes - 1)
            # epoch_segment = 1
            epoch_segment = random.randint(0,num_nodes-1)
            
            visited_nodes = [epoch_segment]

            #그래디언트 초기화 
            output = gat_model(x, adj_tensor)
            #print(output)
            optimizer.zero_grad()

            # Decoder iteration
            visited_nodes, possibilities, rewards, _ = decode_all_dfs(gat_model, output, x, adj_matrix_original, visited_nodes)
            reward_values[graph_idx].append(rewards[-1].item())
                
            # Baseline rollout
            with torch.no_grad():
                visited_nodes = [epoch_segment]
                _, _, baseline_rewards, _ = decode_all_dfs(baseline_model, output, x, adj_matrix_original, visited_nodes)
                # print("baseline_rewards", baseline_rewards)

            # Loss
            loss = custom_loss(rewards, baseline_rewards, possibilities)
            loss_list.append(loss)
            loss_lists[graph_idx].extend(loss_list)
            # if epoch % 500 == 1:
            #     print(f'epoch {epoch} : {loss}')
        
        loss_mean = torch.mean(torch.stack([torch.mean(torch.Tensor(losses)) for losses in loss_lists]))
        reward_mean = torch.mean(torch.stack([torch.mean(torch.Tensor(rewards)) for rewards in reward_values]))

        # Baseline model update
        if epoch % baseline_update_period == 0:
            sync_model(baseline_model, gat_model)
        
        # print(f"loss_mean : {loss_mean}")

        # Backpropagation
        optimizer.zero_grad()
        loss_mean.requires_grad_(True)
        loss_mean.backward(retain_graph=True)
        torch_utils.clip_grad_norm_(gat_model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # 현재 epoch의 loss_mean 값을 저장
        loss_mean_values.append(loss_mean.item())
        reward_mean_values.append(reward_mean.item())
    end_train_time = time.time()
    

    np.save('./log/lossnhnof(8)hfl(' + str(hidden_features) + ')nel(' + str(n_epochs) + ')lr(' + str(lr) + ')do(' + str(dropout) + ')'+'.npy', loss_mean_values)

    np.save('./log/rewardnhnof(8)hfl(' + str(hidden_features) + ')nel(' + str(n_epochs) + ')lr(' + str(lr) + ')do(' + str(dropout) + ')'+'.npy', reward_mean_values)


    model_name = str('gat_model' + "rewards("+ str(sum(reward_mean_values)/len(reward_mean_values)) +')nhnof(8)hfl(' + str(hidden_features) + ')nel(' + str(n_epochs) + ')lr(' + str(lr) + ')do(' + str(dropout) + ')')


    #loss_mean_values를 사용하여 그래프 생성
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, n_epochs+1), loss_mean_values, linestyle='-', linewidth=1)
    plt.xlabel('Epoch')
    plt.ylabel('Loss Mean')
    plt.title('Loss Mean vs. Epoch')
    plt.savefig('./img/' + str(model_name) + '_loss.jpg')
    plt.show(block=False)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, n_epochs+1), reward_mean_values, linestyle='-', linewidth=1)
    plt.xlabel('Epoch')
    plt.ylabel('Reward Mean')
    plt.title('Reward Mean vs. Epoch')
    plt.savefig('./img/' + str(model_name) + '_reward.jpg')
    plt.show(block=False)
    plt.close()


    torch.save(gat_model, './model/' + str(model_name) + '.pt')
    


    return gat_model, end_train_time-start_train_time, model_name