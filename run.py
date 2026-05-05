import torch
import random
import numpy as np

#---------------------------pytorch----------------------------------
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
#----------------------------others----------------------------------
import os
import time
from datetime import datetime
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
import yaml
#--------------------------lib---------------------------------------
from GEN import GraphEditNet
from inference import validation, test
from dataset import get_dataset
import argparse
import time
import wandb

def str2bool(v):
    return v.lower() in ("true", "1")

torch.set_printoptions(precision=4)
gpu = 'cuda:0'
device = torch.device(gpu) if torch.cuda.is_available() \
                              else torch.device("cpu")
                              
def set_seeds_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
def get_loaders(trainset, valset, testset, args):
    val_mask = torch.stack((torch.arange(len(valset)).repeat_interleave(len(trainset)), 
                         torch.arange(len(trainset)).repeat(len(valset)))).t()
    test_mask = torch.stack((torch.arange(len(testset)).repeat_interleave(len(trainset) + len(valset)), 
                            torch.arange(len(trainset) + len(valset)).repeat(len(testset)))).t()
    return val_mask, test_mask

def log_builder(args, run):
    
    record_keys = ['name', 'dataset_name', 'experiment', 'tag']
    comment = ".".join(["{}={}".format(k, v) \
              for k, v in vars(args).items() if k in record_keys])
    current_time = time.strftime("%Y-%m-%dT%H:%M", time.localtime())
    logd = f'./exp_gen_{args.tag}/{args.name}_{args.dataset_name}_{run}_{current_time}'
    logger = SummaryWriter(log_dir = logd, comment = comment)
    config = vars(args)
    config["path"] = logd
    config_file_name = "config.yaml"
    with open(os.path.join(logd, config_file_name), "w") as file:
        file.write(yaml.dump(config))

    return logger, logd

def log_train(logger, loss, num_batches):
    logger.add_scalar(f"loss/train", loss.item(), num_batches)
    
def gen_batch(idx_g1, idx_g2, dataset, args):
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
        source = dataset[idx_g1[i]]
        target = dataset[idx_g2[i]]
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
        geds.append(dataset.ged[source.i, target.i])
    edge_batch = torch.cat(edge_batch)
    
    op_costs = torch.tensor(costs)
    bipartite_edge_index = torch.cat(edge_indices, dim = -1)
    return sources, targets, bipartite_edge_index, op_costs, \
            node_index, edge_batch, geds
     
def train(model, optimizer, trainset, logger, epoch, args):
    model.train()
    total_loss = 0.0
    trainloader = torch.randint(len(trainset),(args.batch_size*args.iterations,2))
    trainloader = iter(DataLoader(trainloader, args.batch_size, shuffle = True))
    if args.loss == 'huber':
        criterion = F.huber_loss
    else:
        criterion = F.mse_loss
    for i in range(args.iterations):
        pairs = next(trainloader)
        start_time = time.time()
        optimizer.zero_grad()
        pairs = pairs.t()
        idx_g1 = pairs[0]
        idx_g2 = pairs[1]
        sources, targets, bipartite_edge_index, op_costs, \
            node_index, edge_batch, geds = gen_batch(idx_g1, idx_g2, trainset, args)
        data = Batch.from_data_list(sources+targets).cuda()
        pred = model(data, bipartite_edge_index.cuda(), op_costs.cuda(), node_index, edge_batch.cuda())
        
        loss = criterion(pred, torch.FloatTensor(geds).to(device).unsqueeze(-1))
        loss.backward()
        optimizer.step()
        log_train(logger, loss, i+args.iterations*(epoch))
        print('Iteration: {:02d}. time: {:.4f}. loss: {:.4f}. '.format(
            i+args.iterations*(epoch), time.time() - start_time, loss.item()),
            end = '                    \r'
            )
        if args.wandb:
            res_dic = {f'loss/train': loss.item(),
                       f'runtime/train': time.time() - start_time}
            wandb.log(res_dic)
        total_loss += loss
    return total_loss / args.iterations


def runGEN(args):
    print(f"executing on {device}")
    results_list = []
    
    for run in range(args.runs):
        set_seeds_all(args.seed[run])
        args.current_seed = args.seed[run]
        print(args)
        logger, logd = log_builder(args, run)
        
        if args.wandb:
            wandb.init(project='GEN',
                       name = f'./exp_{args.name}_{args.dataset_name}_{args.seed[run]}',
                       sync_tensorboard=False)
        
        
        num_node_labels, (graphs, graphs_test), (trainset, valset, testset) = get_dataset(args)
        valloader, testloader = get_loaders(trainset, valset, testset, args)
        if num_node_labels == 0:
            args.input_dim = 1
        else:
            args.input_dim = num_node_labels
        model = GraphEditNet(args)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)
        
        print(f'running repetition {run}')
        bad_epoch = 0
        best_mse = float('inf')
        best_mae = float('inf')
        best_rmse = float('inf')
        best_epoch = -1
        best_kendall = -float('inf')
        best_spearman = -float('inf')
        best = float('inf')
        t0 = time.time()
        
        for epoch in range(args.epochs):
            loss = train(model, optimizer, trainset, logger, epoch, args)
            logger.add_scalar(f'total_loss/train', loss.item(), epoch)
            if (epoch + 1) % args.eval_steps == 0 and (epoch + 1) >= args.eval_interval:
                result = validation(model, valloader, trainset, valset, device, args)
                if result[args.criterion] < best:
                    best_mse = result['mse']
                    best_mae = result['mae']
                    best_rmse = result['rmse']
                    best_epoch = epoch
                    best = result[args.criterion]
                    bad_epoch = 0
                    if args.save_model:
                        path = f'{logd}/{args.name}_{args.dataset_name}_epoch{epoch}_{args.experiment}_criterion={args.criterion}={best:.4f}.pt'
                        torch.save(model.state_dict(), path)
                        path = f'{logd}/{args.name}_{args.dataset_name}_best.pt'
                        torch.save(model.state_dict(), path)
                else:
                    bad_epoch += 1
                    
                logger.add_scalar(f"mse/val", result['mse'], epoch)
                logger.add_scalar(f"mae/val", result['mae'], epoch)
                logger.add_scalar(f"rmse/val", result['rmse'], epoch)
                logger.add_scalar(f"best_mse/val", best_mse, epoch)
                logger.add_scalar(f"best_mae/val", best_mae, epoch)
                logger.add_scalar(f"best_rmse/val", best_rmse, epoch)
                logger.add_scalar(f"best_epoch/val", best_epoch, epoch)
                if args.wandb:
                    res_dic = {f'total_loss/train': loss.item(),
                               f'mse/val': result['mse'],
                               f'mae/val': result['mae'],
                               f'rmse/val': result['rmse'],
                               f'best_mse/val': best_mse,
                               f'best_mae/val': best_mae,
                               f'best_rmse/val': best_rmse,
                               f'best_epoch/val': best_epoch}
                    wandb.log(res_dic)
                print('\nEpoch: {:02d}. Loss: {:.4f}.\n'
                      'Mse: {:.4f}. Best Mse: {:.4f}. \n'
                      'Mae: {:.4f}. Best Mae: {:.4f}. \n'
                      'Rmse: {:.4f}. Best Rmse: {:.4f}. \n'
                      'Best epoch: {:02d}\n'.format(
                        epoch, loss.item(), result['mse'], best_mse, result['mae'], best_mae, result['rmse'], best_rmse, best_epoch), end='               \r')
            if bad_epoch >= args.patience:
                break
                
        path = f'{logd}/{args.name}_{args.dataset_name}_best.pt'
        del model
        model = GraphEditNet(args).cuda()
        model.load_state_dict(torch.load(path))
        result = test(model, testloader, trainset, valset, testset, device, args)
        logger.add_scalar(f"runtime/test", result['runtime'])
        logger.add_scalar(f"mse/test", result['mse'])
        logger.add_scalar(f"mae/test", result['mae'])
        logger.add_scalar(f"rmse/test", result['rmse'])
        logger.add_scalar(f"kendall/test", result['kendall'])
        logger.add_scalar(f"spearman/test", result['spearman'])
        logger.add_scalar(f"p@10/test", result['p10'])
        
        res_dic = {f'runtime/test': result['runtime'],
                   f'mse/test': result['mse'],
                   f'mae/test': result['mae'],
                   f'rmse/test': result['rmse'],
                   f'kendall/test': result['kendall'],
                   f'spearman/test': result['spearman'],
                   f'p@10/test': result['p10']}
        
        if args.wandb:
            wandb.log(res_dic)
        print('\nRun: {:02d}. Runtime: {:.4f}. \n'
              'Best Mse: {:.4f}. Best Mae: {:.4f}. Best Rmse: {:.4f}.\n'
              'Test Mse: {:.4f}. Test Mae: {:.4f}. Test Rmse: {:.4f}.\n'
              'Kendall: {:.4f}. Spearman: {:.4f}. P@10: {:.4f}. \n'
              'Inference time: {:.4f}.\n'.format(
                run, time.time() - t0, best_mse, best_mae, best_rmse, result['mse'], result['mae'], result['rmse'], result['kendall'], result['spearman'], result['p10'], result['runtime']
            ),end = '                    \r')
        logger.add_scalar(f"memory_allocated", torch.cuda.memory_allocated())
        logger.add_scalar(f"memory_cached", torch.cuda.memory_reserved())
        torch.cuda.empty_cache()
        
        filename = f'{logd}/result'
        with open(filename, 'a') as writefile:
            for key, value in result.items():
                writefile.write(key + ' ' + str(value) +'\n')
                    
        result_list = [best_mse, best_mae, best_rmse, result['mse'], result['mae'], result['rmse'], result['kendall'], result['spearman'], result['p10'], result['runtime']]
        path = f'{logd}/result.pt'    
        torch.save(result_list, path)
        
        results_list.append(result_list)
        if args.runs == run + 1:
            best_mse_mean, best_mae_mean, best_rmse_mean, test_mse_mean,\
                test_mae_mean, test_rmse_mean, \
                kendall_mean, spearman_mean, p10_mean, runtime_mean = np.mean(results_list, axis=0)
                
            var = np.var(results_list, axis=0)
            best_mse_std = np.sqrt(var[0])
            best_mae_std = np.sqrt(var[1])
            best_rmse_std = np.sqrt(var[2])
            test_mse_std = np.sqrt(var[3])
            test_mae_std = np.sqrt(var[4])
            test_rmse_std = np.sqrt(var[5])
            kendall_std = np.sqrt(var[6])
            spearman_std = np.sqrt(var[7])
            p10_std = np.sqrt(var[8])
            runtime_std = np.sqrt(var[9])
            final_result = {f'mse_mean/val': best_mse_mean, 
                            f'mse_std/val': best_mse_std,
                            f'mae_mean/val': best_mae_mean, 
                            f'mae_std/val': best_mae_std,
                            f'rmse_mean/val': best_rmse_mean, 
                            f'rmse_std/val': best_rmse_std,
                            f'mse_mean/test': test_mse_mean, 
                            f'mse_std/test': test_mse_std,
                            f'mae_mean/test': test_mae_mean, 
                            f'mae_std/test': test_mae_std,
                            f'rmse_mean/test': test_rmse_mean, 
                            f'rmse_std/test': test_rmse_std,
                            f'kendall_mean/test': kendall_mean, 
                            f'kendall_std/test': kendall_std,
                            f'spearman_mean/test': spearman_mean, 
                            f'spearman_std/test': spearman_std,
                            f'p10_mean/test': p10_mean, 
                            f'p10_std/test': p10_std,
                            f'runtime_mean/test': runtime_mean, 
                            f'runtime_std/test': runtime_std}
            print(final_result)
            if args.wandb:
                wandb.log(final_result)
            filename = f'{logd}/results'
            with open(filename, 'a') as writefile:

                for key, value in final_result.items():
                    writefile.write(key + ' ' + str(value) +'\n')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GEN')
   
    parser.add_argument('--dataset_name', type = str, default = 'aids',
                    help = 'name of the dataset.')
    parser.add_argument('--val_pct', type=float, default=0.2)
    parser.add_argument('--test_pct', type=float, default=0.2)
    parser.add_argument('--experiment', type = str, default = 'ged')
    parser.add_argument('--name', type = str, default = 'GEN')
    parser.add_argument('--seed', type = int, default = [2683, 812, 2306, 6306, 1635],
                        help = 'Random seed number.')
    parser.add_argument('--iterations', type = int, default = 100)                  
    parser.add_argument('--batch_size', type = int, default = 128)
    parser.add_argument('--test_batch_size', type = int, default = 2048)
    parser.add_argument('--epochs', type = int, default = 5000)
    parser.add_argument('--patience', type = int, default = 50)
    parser.add_argument('--save_train', type = str2bool, default = True)
    parser.add_argument('--weight_decay', type = float, default = 0.0)
    parser.add_argument('--lr', type = float, default = 0.001,
                    help = 'Learning rate.')
    parser.add_argument('--runs', type=int, default=5, help='the number of repetition of the experiment to run')
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--eval_interval', type=int, default=10)
    parser.add_argument('--save_model', type=str2bool, default=True)
    parser.add_argument('--wandb', type = str2bool, default = False)
    parser.add_argument('--criterion', type=str, default='mae')
    parser.add_argument('--loss', type=str, default='huber')
    
    parser.add_argument('--cost', type = int, default = [1.0, 1.0, 1.0, 1.0, 1.0])
    parser.add_argument('--hidden_dim', type = int, default = 64)
    parser.add_argument('--output_dim', type = int, default = 32)
    parser.add_argument('--num_layers', type = int, default = 5)
    parser.add_argument('--layer_type', type=str, default='GIN')
    parser.add_argument('--skip_connection', type=str, default='identity')
    parser.add_argument('--activation', type=str, default='ReLU')
    parser.add_argument('--norm', type=str, default='layer')
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--tag', type=str, default='default')
    parser.add_argument('--cost_injection', type = str2bool, default = True)
    parser.add_argument('--prop', type = str2bool, default = True)
    parser.add_argument('--impact', type = str2bool, default = True)
    
    args = parser.parse_args()
    runGEN(args)
