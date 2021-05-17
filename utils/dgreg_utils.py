import os
import torch
import numpy as np
import torch.nn.functional as F
from torch.distributions import Categorical
from scipy import stats

import datasets.utils as datautils

def evaluate_finetune(args, net, epoch, loader):
    row = [epoch]
    row_plcc = [epoch]

    net.eval()
    with torch.no_grad():
        srocc, plcc= evalaute(epoch, net, args.patch_num, loader, cuda=True)
        row.append(srocc)
        row_plcc.append(plcc)

    net.train()
    return row, row_plcc

def add_attri_learn(net):
    for name, param in net.named_parameters():
        param.learn = False if 'rep' in name else True

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def select_action(remaining_scores):
    probs = F.softmax(remaining_scores.view(-1), dim=0)
    m = Categorical(probs)
    action = m.sample()
    log_prob = m.log_prob(action).view(1)
    return action.item(), log_prob

def cal_srocc(pred, target):
    '''
    return SROCC correlation -1. ~ 1.
    '''
    return stats.spearmanr(pred, target)[0]

def cal_plcc(pred, target):
    '''
    return SROCC correlation -1. ~ 1.
    '''
    return stats.pearsonr(pred, target)[0]

def cal_policy_loss(reward_pool, logprob_pool, gamma):
    ##  Discount reward
    running_add = 0
    length = len(reward_pool)
    for i in reversed(range(length)):
        if reward_pool[i] == 0:
            running_add = 0
        else:
            running_add = running_add * gamma + reward_pool[i]
            reward_pool[i] = running_add

    # Normalize reward
    reward_mean = np.mean(reward_pool)
    reward_std = np.std(reward_pool)
    for i in range(length):
        reward_pool[i] = (reward_pool[i] - reward_mean) / (reward_std + 1e-6)

    policy_loss = 0
    for i in range(length):
        policy_loss += -logprob_pool[i] * reward_pool[i]  # Negtive score function x reward

    return policy_loss


def save_checkpoint(step, model, folder, name):
    folder = os.path.join(folder, name)
    if not os.path.exists(folder):
        # os.mkdir(folder)
        os.makedirs(folder)
    model_out_path = folder+'/'+"{}_step_{}.pth".format(name, step)
    # model_out_path = folder + '/' + 'best.pth'
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint best {} saved to {}".format(step, model_out_path))


def write_csv(csv_file, data):
    with open(csv_file, 'w', newline='') as t_file:
       csv_writer = csv.writer(t_file)
       for l in data:
           csv_writer.writerow(l)

def evalaute(epoch, net, patch_num, loader, cuda=True):

    gt_mos = torch.Tensor([])
    pred_score = torch.Tensor([])

    for i, [demo, mos] in enumerate(loader):
        # demo:(bs,3,224,224), mos:(bs,1)
        mos = mos.unsqueeze(-1)
        gt_mos = torch.cat([gt_mos, mos], 0)

        if cuda:
            demo = demo.cuda()

        with torch.no_grad():
            score, _, _, _ = net(demo)

        pred_score = torch.cat([pred_score, score.cpu()], 0)

    pred_score = np.mean(np.reshape(np.array(pred_score[:,0]), (-1, patch_num)), axis=1)
    gt_mos = np.mean(np.reshape(np.array(gt_mos[:,0]), (-1, patch_num)), axis=1)
    try:
        return cal_srocc(pred_score[:,0], gt_mos[:,0]), cal_plcc(pred_score[:,0], gt_mos[:,0])
    except:
        return cal_srocc(pred_score, gt_mos), cal_plcc(pred_score, gt_mos)




def cal_similarity(x, p=2, dim=1):
    '''
    x: (n,K)
    return: (n,n)
    '''
    x = F.normalize(x, p=p, dim=dim)
    return torch.mm(x, x.transpose(0, 1))

def cal_edge_emb(x, p=2, dim=1):
    '''
    x: (n,K)
    return: (n^2, K)
    '''
    x = F.normalize(x, p=p, dim=dim)
    x_r = torch.transpose(x, 0, 1).unsqueeze(2)  # (K, n, 1)
    x_c = torch.transpose(x, 0, 1).unsqueeze(1)  # (K, 1, n)
    A = torch.bmm(x_r, x_c).permute(1,2,0)  # (n, n, K)

    A = A.view(A.size(0) * A.size(1), A.size(2))  # (n^2, K)
    # print(A.size())
    return A.cuda()

def first_order_filter(x):
    h_x = F.pad(x, pad=(0,1,0,0), mode='constant', value=0)[None, None, :]    # pad: (left, right, top, down)
    h_kernel = torch.Tensor([1,-1])[None, None, None, :].cuda()    # (1,1,1,2)
    h_x = F.conv2d(h_x, weight=h_kernel)
    h_x = torch.triu(h_x, diagonal=1)

    v_x = F.pad(x, pad=(0,0,1,0), mode='constant', value=0)[None, None, :]
    v_kernel = torch.Tensor([[-1],[1]])[None, None, :].cuda()      # (1,1,2,1)
    v_x = F.conv2d(v_x, weight=v_kernel)
    v_x = torch.triu(v_x, diagonal=1)

    return h_x, v_x


def graph_norm(A, batch=False, self_loop=True, symmetric=True):
	# A = A + I    A: (bs, num_nodes, num_nodes)
    if self_loop:
        eye = torch.eye(A.size(1)).expand(A.size())
        eye = eye.cuda() if A.is_cuda else eye
        A = A + eye

    # Degree
    d = A.sum(-1) # (bs, num_nodes)
    if symmetric:
		# D = D^-1/2
        d = torch.pow(d, -0.5)
        if batch:
            D = A.detach().clone()
            for i in range(A.size(0)):
                D[i] = torch.diag(d[i])
            norm_A = D.bmm(A).bmm(D)
        else:
            D = torch.diag(d)
            norm_A = D.mm(A).mm(D)
    else:
		# D=D^-1
        d = torch.pow(d,-1)
        if batch:
            D = A.detach().clone()
            for i in A.size(0):
                D[i] = torch.diag(d[i])
            norm_A = D.bmm(A)
        else:
            D =torch.diag(d)
            norm_A = D.mm(A)

    return norm_A


def fetch_batch(loaders, iters, task=None, num=1, iflist=True):
    task = str(task)
    x_list, y_list, level_list = [], [], []
    for _ in range(num):
        try:
            x, y, level = next(iters[task]) # (bs,c,h,w), (bs,1)
        except:
            iters[task] = iter(loaders[task])
            x, y, level = next(iters[task])
        x_list.append(x)
        y_list.append(y)
        level_list.append(level)
        if num == 1 and not iflist:
            return x_list[0], y_list[0], level_list[0]
        else:
            return x_list, y_list, level_list




