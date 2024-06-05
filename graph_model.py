
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import copy
device = torch.device('cuda')
# Encoder
class GraphAttentionLayer(torch.nn.Module):
    def __init__(self, in_features, out_features, n_heads, is_concat = True, dropout = 0.6, leacky_relu_negative_slope = 0.2):
        super(GraphAttentionLayer, self).__init__()
        self.W = torch.nn.Parameter(torch.randn(in_features, out_features))
        self.is_concat = is_concat
        self.n_heads = n_heads

        if is_concat:
            assert out_features % n_heads == 0

            self.n_hidden = out_features // n_heads
        else:
            self.n_hidden = out_features

        self.linear = nn.Linear(in_features, self.n_hidden * n_heads, bias = False)

        self.attn = nn.Linear(self.n_hidden * 2, 1, bias = False)
        self.activation = nn.LeakyReLU(negative_slope = leacky_relu_negative_slope)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout) 

    def forward(self, x, adj):
        n_nodes = x.shape[0]
        g=self.linear(x).view(n_nodes, self.n_heads, self.n_hidden)
        g_repeat = g.repeat(n_nodes, 1,1) 
        g_repeat_interleave = g.repeat_interleave(n_nodes, dim=0)
        g_concat = torch.cat([g_repeat_interleave, g_repeat], dim = -1)
        g_concat = g_concat.view(n_nodes, n_nodes, self.n_heads, 2 * self.n_hidden)
        e = self.activation(self.attn(g_concat))
        e = e.squeeze(-1)
        adj = adj.repeat(1, 1, self.n_heads)
        assert adj.shape[0] == 1 or adj.shape[0] == n_nodes
        assert adj.shape[1] == 1 or adj.shape[1] == n_nodes
        assert adj.shape[2] == 1 or adj.shape[2] == self.n_heads
        e=e.masked_fill(adj == 0, 1)
        a = self.softmax(e)
        a = self.dropout(a)
        attn_res = torch.einsum('ijh,jhf->ihf', a, g)
        if self.is_concat:
            return attn_res.reshape(n_nodes, self.n_heads * self.n_hidden)
        else:
            attn_res = attn_res.mean(dim=1)
            return attn_res

class FeedForward(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout=0.6):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(in_features, hidden_features)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_features, out_features)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.dropout1(self.linear1(x)))
        x = self.dropout2(self.linear2(x))
        return x
# Decoder
class Decoder(torch.nn.Module):
    def __init__(self, in_features, hidden_features, out_features, n_heads, d_h):
        super(Decoder, self).__init__()
        self.n_heads = n_heads
        self.hidden_features = hidden_features
        self.d_h = d_h
        self.in_features = in_features

        self.phi1 = torch.nn.Linear(self.n_heads, self.hidden_features)
        self.phi2 = torch.nn.Linear(self.n_heads, self.hidden_features)
        self.softmax = nn.Softmax(dim=0)
        self.C = 10     # constant C
        self.activation = nn.Tanh()

    def forward(self, output: torch.Tensor, prev_node: int, adj_modified: torch.Tensor):
        v_i = output[prev_node]     # d_h
        v_j = adj_modified     # N
        v_j = v_j.unsqueeze(1).cuda() * output      # N x d_h
        
        v_i = v_i.unsqueeze(0)
        phi1_v_i = self.phi1(v_i)
        phi2_v_j = self.phi2(v_j)

        attn_input = torch.matmul(phi1_v_i, phi2_v_j.transpose(0, 1)) / (self.d_h ** 0.5)   # 1 x N

        attn_output = self.C * self.activation(attn_input)      # 1 x N

        attn_output = attn_output * adj_modified.to(device)     # 1 x N
        attn_output = attn_output.squeeze()

        attn_weights = self.softmax(attn_output)
        attn_weights = attn_weights * (attn_output != 0).float()

        p = attn_weights.max()
        selected_node = torch.argmax(attn_weights).item()
        return selected_node, p


class GAT(torch.nn.Module):
    def __init__(self, in_features, hidden_features, out_features, n_heads, d_h, dropout):
        super(GAT, self).__init__()
        self.n_heads = n_heads
        self.attention1 = GraphAttentionLayer(in_features, hidden_features, n_heads, is_concat = True, dropout = dropout)
        self.attention2 = GraphAttentionLayer(hidden_features, out_features, 1, is_concat = False, dropout = dropout)
        self.feed_forward = FeedForward(out_features, hidden_features, out_features, dropout)
        self.decoder = Decoder(in_features, hidden_features, out_features, n_heads, d_h)
    
    def forward(self, x, adj):
        x = self.attention1(x, adj)
        x = self.attention2(x, adj)
        x = self.feed_forward(x)
        x = F.softmax(x, dim=0)
        return x
    
    def decode(self, output, i, adj_modified):
        return self.decoder(output, i, adj_modified)
    





def sync_model(target_model, source_model):
    target_model.load_state_dict(source_model.state_dict())

def decode_all_dfs(model, encoder_output, x, adj_matrix_original, visited_nodes):
    n = encoder_output.size(0)

    branch_point = []
    previous_nodes = []
    possibilities = torch.empty(0).cuda()

    rewards = []
    accumulated_w = np.zeros(n)
    accumulated_w[visited_nodes[0]] = 1.0

    for i in range(n-1):
        prev_node = visited_nodes[i]
        row_indices = np.where(adj_matrix_original[:,prev_node] == 1)[0]
        row_indices = np.setdiff1d(row_indices, visited_nodes)

        if len(row_indices) >= 2:
            branch_point.append(prev_node)
        elif len(row_indices) == 0:
            while len(row_indices) == 0:
                if len(branch_point) == 0:
                    break
                prev_node = branch_point.pop()
                row_indices = np.where(adj_matrix_original[:,prev_node] == 1)[0]
                row_indices = np.setdiff1d(row_indices, visited_nodes)
                
                if len(row_indices) >= 2:
                    branch_point.append(prev_node)

        previous_nodes.append(prev_node)

        adj_modified = adj_matrix_original.clone()
        adj_modified[row_indices, :] = 1

        other_indices = [i for i in range(adj_matrix_original.shape[0]) if i not in row_indices]
        adj_modified[other_indices, :] = 0
        adj_modified = adj_modified[:, 1]

        selected_node, p = model.decode(encoder_output, prev_node, adj_modified)

        visited_nodes.append(selected_node)
        possibilities = torch.cat([possibilities, p.unsqueeze(0)], dim=0)
        accumulated_w[selected_node] = accumulated_w[prev_node] + x[selected_node].item()

        # calculate reward
        reward = accumulated_w.sum()
        rewards.append(reward)
    rewards = torch.Tensor(rewards)

    return visited_nodes, possibilities, rewards, previous_nodes

def custom_loss(rewards, baseline_rewards, possibilities):
    rewards = rewards.cuda()
    baseline_rewards = baseline_rewards.cuda()
    possibilities = possibilities.cuda()
    return ((rewards - baseline_rewards) * torch.log(possibilities)).sum()