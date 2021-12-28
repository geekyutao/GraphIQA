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
import random

from models.networks import Reg_Domain



def main(args):
    # HYPER-PARAMETERS
    STEPS = args.steps
    lr = args.lr
    batch_size = args.bs

    start_step = args.restore_epoch if args.restore else 1

    # System
    utils.set_seed(args.seed)
    gpus_list = list(range(args.gpus))
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Build DataLoader and Iterator
    folder_path = {
        'live': '/data2/sunsm/Datasets/databaserelease2',
        'csiq': './data/csiq',
        'tid2013': './data/TID2013',
        'livec': './data/ChallengeDB',
        'koniq': './data/KonIQ',
        'bid': './data/BID',
        'kadid-F': './data/kadid10k'
    }

    dataset_path = folder_path[args.dataset]

    img_num = {
        'live': list(range(0, 29)),
        'csiq': list(range(0, 30)),
        'tid2013': list(range(0, 25)),
        'livec': list(range(0, 1162)),
        'koniq': list(range(0, 10073)),
        'bid': list(range(0, 586)),
        'kadid-F':list(range(0,81))
    }
    sel_num = img_num[args.dataset]
    # ratio = 0.8


    random.shuffle(sel_num)


    train_index = sel_num[0:int(round(0.8 * len(sel_num)))]
    test_index = sel_num[int(round(0.8 * len(sel_num))):len(sel_num)]

    # tb
    if args.tb:
        tb_name = os.path.join('./tb/', args.model_name)
        writer = SummaryWriter(log_dir=tb_name)

    # Model
    net = Reg_Domain(do_emb_size=args.dosz, eg_emb_size=args.egsz, pretrain=False)
    # print(net)
    net = net.to(device)
    if args.gpus > 1:
        net = torch.nn.DataParallel(net, device_ids=gpus_list)

    if args.restore:
        model_name = os.path.join(args.ckpt)
        print('pretrained model: %s' % model_name)
        if os.path.exists(model_name):
            pretained_model = torch.load(model_name)
            model_dict = net.state_dict()
            state_dict = {k: v for k, v in pretained_model.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            net.load_state_dict(model_dict)
        else:
            raise Exception("Checkpoint Not Found!")

    optimizer = optim.Adam(net.parameters(), lr=lr)
    MSE_loss = torch.nn.MSELoss()

    train_loader = datautils.make_dataloader(
        dataset_name=args.dataset,
        dataset_path=dataset_path,
        csv_path=None,
        task_list=None,
        level_list=None,
        bs=batch_size,
        shuffle=True,
        num_workers=args.gpus,
        drop_last=True,
        mode='all',
        trainsz=train_index, patch_num=args.patch_num,sel='all')

    if args.dataset=='csiq':
        test_sel = ['all', 'AWGN', 'BLUR', 'contrast','fnoise','JPEG','jpeg2000']
    elif args.dataset=='live':
        test_sel = ['all', 'jp2k', 'jpeg', 'wn', 'gblur', 'fastfading']
    elif (args.dataset=='koniq') | (args.dataset=='livec'):
        test_sel = ['all']
    elif args.dataset=='kadid':
        test_sel = ['all',1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]

    test_loaders = {}
    test_sel = ['all']
    for sel in test_sel:
        test_loader = datautils.make_dataloader(
            dataset_name=args.dataset,
            dataset_path=dataset_path,
            csv_path=None,
            task_list=None,
            level_list=None,
            bs=1,
            shuffle=False,
            num_workers=args.gpus,
            drop_last=True,
            mode='all',
            trainsz=test_index, patch_num=args.patch_num, sel=sel)
        # test_iterator = iter(test_loader)
        test_loaders[sel] = test_loader


    print('------------------------------------------no pretrain test-------------------------------------------')
    for sel in test_sel:
        if sel == 'all':
            eval_row, eval_row_plcc = utils.evaluate_finetune(args, net, 1,
                                                              loader=test_loaders['all'])
            test_avg_srocc = eval_row[-1]
            test_avg_plcc = eval_row_plcc[-1]
            print('BEST: Avg SROCC: {:.4f}, Avg PLCC: {:.4f}'.format(test_avg_srocc, test_avg_plcc))
        else:
            type_srcc_row, type_plcc_row = utils.evaluate_finetune(args, net, 1,
                                                                   loader=test_loaders[sel])
            type_srocc = type_srcc_row[-1]
            type_plcc = type_plcc_row[-1]
            print('| {}: Avg SROCC: {:.4f}, Avg PLCC: {:.4f}'.format(sel, type_srocc, type_plcc))

    print('-----------------------------------------------------------------------------------------------------')


    # # Train
    ma_loss, ma_srocc, best_srocc, ma_plcc = 0, 0, 0, 0
    for s in range(start_step, STEPS):
        count = 0
        for j, [anc_x, anc_y] in enumerate(train_loader):
            # anc_x:(bs,3,224,224), anc_y:(bs,)
            anc_y = anc_y.unsqueeze(-1)  # anc_y:(bs,1)

            # Forward
            anc_x, anc_y = anc_x.to(device), anc_y.to(device)
            pred_y, _, _, _ = net(anc_x)

            # Loss
            loss = MSE_loss(pred_y, anc_y)

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # log
            loss_scalar = loss.detach().item()
            srocc = utils.cal_srocc(pred_y[:,0].detach().cpu(), anc_y[:,0].detach().cpu())
            plcc = utils.cal_plcc(pred_y[:,0].detach().cpu(), anc_y[:,0].detach().cpu())
            ma_loss += loss_scalar
            ma_srocc += srocc
            ma_plcc += plcc

            count += batch_size

        if args.tb:
            writer.add_scalar('SROCC', ma_srocc / count, s)
            writer.add_scalar('PLCC', ma_plcc / count, s)
            writer.add_scalar('Total loss', ma_loss / count, s)

        # Test
        eval_row, eval_row_plcc = utils.evaluate_finetune(args, net, s, loader=test_loaders['all'])
        test_avg_srocc = eval_row[-1]
        test_avg_plcc = eval_row_plcc[-1]

        if test_avg_srocc > best_srocc:
            best_srocc = test_avg_srocc
            print('BEST: Avg SROCC: {:.4f}, Avg PLCC: {:.4f}'.format(test_avg_srocc, test_avg_plcc))
            utils.save_checkpoint(s, net, './ckpt', args.model_name)
            for sel in test_sel:
                if sel =='all':
                    continue
                else:
                    type_srcc_row, type_plcc_row = utils.evaluate_finetune(args, net, s,
                                                                      loader=test_loaders[sel])
                    type_srocc = type_srcc_row[-1]
                    type_plcc = type_plcc_row[-1]
                    print('| {}: Avg SROCC: {:.4f}, Avg PLCC: {:.4f}'.format(sel, type_srocc, type_plcc))


        if args.tb:
            writer.close()





if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    # model
    argparser.add_argument('--dataset', help='Dataset: kadid-F/live/csiq/livec/koniq', default='live')
    argparser.add_argument('--model_name', help='Name of model to be saved', default='')
    argparser.add_argument("--tb", action="store_true", default=False)
    argparser.add_argument('--restore', help='Use a checkpoint model', action='store_true', default=False)
    argparser.add_argument('--restore_epoch', type=int, help='restorefrom which epoch', default=1)
    argparser.add_argument('--ckpt', help='Path to checkpoint', default='')
    argparser.add_argument('--gpus', default=1, type=int, help='number of gpu')
    argparser.add_argument('--gpu_id', type=str, default='2', help='GPU ID')

    # RL setting
    argparser.add_argument('--lr', type=float, help='learning rate', default=5e-6)
    argparser.add_argument('--steps', default=50, type=int, help='How many episodes')
    argparser.add_argument('--bs', default=32, type=int, help='How many episodes to update policy')
    argparser.add_argument('--patch_num', default=25, type=int, help='How many patches to crop randomly')

    # general setting
    argparser.add_argument('--seed', type=int, help='Seed for random', default=10000)
    argparser.add_argument('--print_inter', default=100, type=int, help='How many steps to print info')
    argparser.add_argument('--eval_inter', default=1, type=int, help='How many steps to evalaute model')
    argparser.add_argument('--save_inter', default=1000, type=int, help='How many steps to save model')
    argparser.add_argument('--simi_loss', type=float, help='weight of similarity matrix loss', default=0)
    argparser.add_argument('--trisz', default=1, type=int, help='Number of triplet candicates')
    argparser.add_argument('--reg_w', type=float, help='weight of Regression loss', default=1)
    argparser.add_argument('--cls_w', type=float, help='weight of Regression loss', default=0.5)
    argparser.add_argument('--do_w', type=float, help='weight of domain triplet loss', default=0.5)

    argparser.add_argument('--dosz', default=256, type=int, help='Domain node embedding size')
    argparser.add_argument('--egsz', default=64, type=int, help='Domain node embedding size')


    args = argparser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    print(args)

    main(args)
