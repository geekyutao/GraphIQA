import datasets.utils as datautils
import utils.dgreg_utils as utils
from tensorboardX import SummaryWriter
import torch
from torch import optim
import torch.nn.functional as F
import argparse
import csv
import os
import numpy as np
from copy import deepcopy

from models.networks import Reg_Domain



def main(args):
    # HYPER-PARAMETERS
    STEPS = args.steps
    lr = args.lr
    batch_size = args.bs
    print_interval = args.print_inter
    eval_interval = args.eval_inter
    start_step = args.restore_epoch if args.restore else 1
    if args.dataset == 'kadid-P':
        all_tasks = list(range(1,26))   # These are all 25 tasks from kadid10k
    else:
        all_tasks = list(range(1,26))
        del all_tasks[7]
    train_tasks = all_tasks
    train_levels = list(range(1,6)) # 1 for the best quality, 5 for the worst
    triplet_size = args.trisz

    # System
    utils.set_seed(args.seed)
    gpus_list = list(range(args.gpus))
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Recording
    csv_path = './csv/'
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)
    csvfile = csv_path + args.model_name + '.csv'
    with open(csvfile, 'w') as csv_file:
        csvwriter = csv.writer(csv_file)
        csvwriter.writerow(["Step"]+["Train_Avg_SROCC"]+["Test_Avg_SROCC"])


    if args.tb:
        tb_name = os.path.join('./tb/', args.model_name)
        writer = SummaryWriter(log_dir=tb_name)


    # Build DataLoader and Iterator
    print("Build all training task dataloader & iterator dict...")
    train_loaders, train_iterators = {}, {}
    for t in train_tasks:
        loader = datautils.make_dataloader(
            dataset_name=args.dataset,
            dataset_path=args.dataset_path,
            csv_path=args.csv_path,
            task_list=[t],
            level_list=train_levels,
            bs=batch_size,
            shuffle=True,
            num_workers=args.gpus,
            drop_last=True,
            mode='all',
            trainsz=None)
        iterator = iter(loader)

        t = str(t)
        train_loaders[t] = loader
        train_iterators[t] = iterator

    # Model
    net = Reg_Domain(do_emb_size=args.dosz, eg_emb_size=args.egsz, pretrain=True)
    net = net.to(device)
    if args.gpus > 1:
        net = torch.nn.DataParallel(net, device_ids=gpus_list)

    if args.restore:
        model_name = os.path.join(args.ckpt)
        print('pretrained model: %s' % model_name)
        if os.path.exists(model_name):
            pretained_model = torch.load(model_name)
            net.load_state_dict(pretained_model)
        else:
            raise Exception("Checkpoint Not Found!")

    optimizer = optim.Adam(net.parameters(), lr=lr)
    MSE_loss = torch.nn.MSELoss()
    cross_entropy_loss = torch.nn.CrossEntropyLoss()

    # Train
    correct = 0
    ma_loss, ma_mse_loss, ma_tri_loss, ma_srocc, best_srocc = 0, 0, 0, 0, 0
    for s in range(start_step, STEPS):

        # Fetch a batch
        task_idx = np.random.choice(len(train_tasks), 1)[0]
        task = train_tasks[task_idx]
        neg_tasks = deepcopy(train_tasks)
        del neg_tasks[task_idx]
        neg_tasks = np.random.choice(neg_tasks, triplet_size, replace=True)

        anc_x, anc_y, anc_level = utils.fetch_batch(train_loaders, train_iterators, task, 1, iflist=False)

        # Sample a batch of triplets
        pos_xs, pos_ys, _ = utils.fetch_batch(train_loaders, train_iterators, task, triplet_size, iflist=True)
        neg_xs, neg_ys = [], []
        for i in range(triplet_size):
            neg_x, neg_y, _ = utils.fetch_batch(train_loaders, train_iterators, neg_tasks[i], 1, iflist=False)
            neg_xs.append(neg_x), neg_ys.append(neg_y)

        # Forward
        anc_x, anc_y = anc_x.to(device), anc_y.to(device)
        # pred, do_emb, do_code = net(anc_x)
        _ , do_code, level_pred, type_pred = net(anc_x)

        with torch.no_grad():
            pos_x, pos_y, neg_x, neg_y = pos_xs[0].to(device), pos_ys[0].to(device),\
                neg_xs[0].to(device), neg_ys[0].to(device)
            _, pos_do_code, _, pos_type_pred = net(pos_x)
            _, neg_do_code, _, neg_type_pred = net(neg_x)


        anc_level = anc_level.to(device)

        cls_loss = MSE_loss(level_pred, anc_level)

        # # soft margin
        # d_pos = torch.dist(do_code, pos_do_code, p=2)
        # d_neg = torch.dist(do_code, neg_do_code, p=2)
        # diff = d_pos - d_neg
        # soft_margin = F.softplus(diff).detach().cpu().float()

        # Loss
        tri_loss = torch.nn.TripletMarginLoss(margin=args.margin)
        triplet_loss = tri_loss(do_code[None], pos_do_code[None], neg_do_code[None])

        loss = args.do_w * triplet_loss + args.cls_w * cls_loss

        # Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # log
        mse_loss_scalar = cls_loss.detach().item()
        tri_loss_scalar = triplet_loss.detach().item()
        loss_scalar = loss.detach().item()
        ma_mse_loss += mse_loss_scalar
        ma_tri_loss += tri_loss_scalar
        ma_loss += loss_scalar


        if args.tb:
            writer.add_scalar('Total loss', loss, s)
            writer.add_scalar('Level loss', cls_loss, s)
            writer.add_scalar('Triplet loss', triplet_loss, s)

        if s % print_interval == 0:
            print(">> Step {}-{} |Avg loss: {:.4f} Cls loss: {:.4f} Tri loss: {:.4f}"\
            .format(s-print_interval, s, ma_loss/print_interval, ma_mse_loss/print_interval, ma_tri_loss/print_interval))
            ma_loss, ma_srocc, ma_mse_loss, ma_tri_loss = 0, 0, 0, 0

        if s % args.save_inter == 0:
            utils.save_checkpoint(s, net, './ckpt', args.model_name)
        print('debug')


    if args.tb:
        writer.close()

    csvfile.close()




if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    # model
    argparser.add_argument('--dataset', help='Dataset: Kadid, ChallengeDB or KonIQ', default='kadis-P')
    argparser.add_argument('--model_name', help='Name of model to be saved', default='Pretrain-model')
    argparser.add_argument("--tb", action="store_true", default=True)
    argparser.add_argument('--restore', help='Use a checkpoint model', action='store_true', default=False)
    argparser.add_argument('--restore_epoch', type=int, help='restorefrom which epoch', default=1)
    argparser.add_argument('--ckpt', help='Path to checkpoint',
                           default='')
    argparser.add_argument('--gpus', default=1, type=int, help='number of gpu')
    argparser.add_argument('--gpu_id', type=str, default='6', help='GPU ID')

    # RL setting
    argparser.add_argument('--lr', type=float, help='learning rate', default=1e-5)
    argparser.add_argument('--steps', default=10000000, type=int, help='How many episodes')
    argparser.add_argument('--bs', default=32, type=int, help='How many episodes to update policy')

    # general setting
    argparser.add_argument('--seed', type=int, help='Seed for random', default=10000)
    argparser.add_argument('--csv_path', help='file list of meta-test dataset', default='./data/kadis700k/25_types_csv')
    argparser.add_argument('--dataset_path', help='Path to dataset', default='./data/kadis700k/images')
    argparser.add_argument('--print_inter', default=100, type=int, help='How many steps to print info')
    argparser.add_argument('--eval_inter', default=10000000, type=int, help='How many steps to evalaute model')
    argparser.add_argument('--save_inter', default=5000, type=int, help='How many steps to save model')
    argparser.add_argument('--simi_loss', type=float, help='weight of similarity matrix loss', default=0)
    argparser.add_argument('--trisz', default=1, type=int, help='Number of triplet candicates')
    argparser.add_argument('--reg_w', type=float, help='weight of Regression loss', default=1)
    argparser.add_argument('--cls_w', type=float, help='weight of Regression loss', default=0.25)
    argparser.add_argument('--do_w', type=float, help='weight of domain triplet loss', default=1)
    argparser.add_argument('--in_w', type=float, help='weight of instance contrastive loss', default=1)
    argparser.add_argument('--simi_w', type=float, help='weight of similarity matrix loss', default=0.01)
    argparser.add_argument('--aggre', help='How to aggregate domain and instance info: concat or bilinear', default='concat')
    argparser.add_argument('--dosz', default=256, type=int, help='Domain node embedding size')
    argparser.add_argument('--egsz', default=64, type=int, help='Domain node embedding size')
    argparser.add_argument('--insz', default=256, type=int, help='Instance node embedding size')
    argparser.add_argument('--margin', type=float, help='margin', default=0.1) # 0.1 or 1

    args = argparser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    print(args)

    main(args)
