import torch
import numpy as np
import random
import time
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error
from scipy.stats import kendalltau, spearmanr
from torch_geometric.data import Batch

def gen_batch_for_test(idx_g1, idx_g2, s, t, args):
    sources = []
    targets = []
    geds = []
    geds = []
    node_index = 0
    costs = []
    graph_index = 0
    edge_indices = []
    edge_weights = []
    edge_batch = []
    for i in range(len(idx_g1)):
        source = s[idx_g1[i]]
        target = t[idx_g2[i]]
        max_nodes = max(source.num_nodes, target.num_nodes)
        if source.x == None:
            source.x = torch.tensor(source.num_nodes * [1.0]).unsqueeze(-1)
            target.x = torch.tensor(target.num_nodes * [1.0]).unsqueeze(-1)
        bipartite_edge_index = torch.stack((torch.repeat_interleave(torch.arange(node_index, node_index + max_nodes),max_nodes),
                                             torch.arange(node_index, node_index + max_nodes).repeat(max_nodes)))
        edge_batch.append(torch.tensor(graph_index).repeat(bipartite_edge_index.size(1)))
        edge_indices.append(bipartite_edge_index)
        graph_index += 1
        node_index += max_nodes
        
        source.x = torch.cat((source.x, torch.tensor([0.0]*source.x.size(1)).repeat(max_nodes-source.x.size(0),1)))
        target.x = torch.cat((target.x, torch.tensor([0.0]*target.x.size(1)).repeat(max_nodes-target.x.size(0),1)))
        source.num_nodes = source.x.size(0)
        target.num_nodes = target.x.size(0)
        sources.append(source)
        targets.append(target)
        costs.append(args.cost)
        geds.append(s.ged[source.i, target.i])
    edge_batch = torch.cat(edge_batch)
    
    op_costs = torch.tensor(costs)
    bipartite_edge_index = torch.cat(edge_indices, dim = -1)
    return sources, targets, bipartite_edge_index, op_costs, \
            node_index, edge_batch, geds
            
def _cal_p_at_k(k, gt):
    gt_inc = np.sort(gt)
    tmp = (gt_inc <= gt_inc[k-1]).sum()
    if tmp > k:
        best_k_gt = gt.argsort()[:tmp]
    else:
        best_k_gt = gt.argsort()[:k]
    return best_k_gt

def cal_p_at_k(k, gt, r_pred):
    best_k_pred = r_pred[::-1][:k]
    best_k_gt = _cal_p_at_k(k, -gt)
    return len(set(best_k_pred).intersection(set(best_k_gt))) / k
    
def evaluator(pred, gt, mat, mat_gt, test):
    mse = mean_squared_error(gt, pred)
    mae = mean_absolute_error(gt, pred)
    rmse = root_mean_squared_error(gt, pred)
    kendall_list = []
    spearman_list = []
    p10_list = []
    kendall = 0
    spearman = 0
    p10 = 0
    if test:
        for pred, gt in zip(mat, mat_gt):
            tmp_pred = pred.argsort()
            r_pred = np.empty_like(tmp_pred)
            r_pred[tmp_pred] = np.arange(len(pred))
            
            tmp_gt = gt.argsort()
            r_gt = np.empty_like(tmp_gt)
            r_gt[tmp_gt] = np.arange(len(gt))
            kendall_list.append(kendalltau(r_pred, r_gt).statistic)
            spearman_list.append(spearmanr(r_pred, r_gt).statistic)
            p10_list.append(cal_p_at_k(10, gt, tmp_pred))
        kendall = np.mean(kendall_list).item()
        spearman = np.mean(spearman_list).item()
        p10 = np.mean(p10_list).item()
    return (mse, mae, rmse, kendall, spearman, p10)

@torch.no_grad()
def get_preds(model, loader, s, t, test, device, args):
    preds = []
    gt = []
    t0 = time.time()
    
    for pairs in loader:
        tasks = []
        start_time = time.time()
        pairs = pairs.t()
        idx_g1 = pairs[0]
        idx_g2 = pairs[1]
        sources, targets, bipartite_edge_index, op_costs, \
            node_index, edge_batch, geds = gen_batch_for_test(idx_g1, idx_g2, s, t, args)
        data = Batch.from_data_list(sources+targets).cuda()
        pred = model(data, bipartite_edge_index.cuda(), op_costs.cuda(), node_index, edge_batch.cuda())
        preds.append(pred.squeeze())
        gt.append(torch.FloatTensor(geds))
    runtime = time.time() - t0
    mse, mae = 0.0, 0.0
    pred = torch.cat(preds, dim = -1).detach().cpu().numpy()
    gt = torch.cat(gt, dim = -1).detach().cpu().numpy()
    mat = None
    mat_gt = None
    if test:
        mat = pred.reshape(len(s),len(t))
        mat_gt = gt.reshape(len(s),len(t))
    (mse, mae, rmse, kendall, spearman, p10) = evaluator(pred, gt, mat, mat_gt, test)
    result = {'mse': mse,
               'mae': mae,
               'rmse': rmse,
               'kendall': kendall,
               'spearman': spearman,
               'p10': p10,
               'runtime': runtime}
    return result

@torch.no_grad()
def test(model, testloader, trainset, valset, testset, device, args):
    #print('starting testing')
    model.eval()
    testloader = iter(DataLoader(testloader, args.test_batch_size, shuffle = False))

    result = get_preds(model, testloader, testset, trainset+valset, True, device, args)
    return result

@torch.no_grad()
def validation(model, valloader, trainset, valset, device, args):
    #print('starting testing')
    model.eval()
    
    valloader = iter(DataLoader(valloader, args.batch_size, shuffle = True))
    result = get_preds(model, valloader, valset, trainset, False, device, args)
    return result
