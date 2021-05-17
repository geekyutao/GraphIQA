import os
import csv
import torch
import pandas as pd
from .kadid import Kadid10k
from .folders import *


def make_dataloader(dataset_name, dataset_path, csv_path, task_list, level_list, \
    mode, trainsz, bs, shuffle=True, num_workers=1, drop_last=False, patch_num=25, sel='all'):
    '''
    level_list: which distortion levels to choose
    mode: 'spt', 'qry', 'all'
    trainsz: if 'train', how large is the training size
    '''

    if dataset_name in ['kadid-P', 'kadis-P']:
        dataset_dict = {'img': [], 'mos': [], 'level': []}
        for t in task_list:
            csv_name = 'type'+str(t)+'.csv'
            csv_file = os.path.join(csv_path, csv_name)
            df = pd.read_csv(csv_file)
            dataset_dict['img'] = dataset_dict['img'] + df['dist_img'].tolist()
            dataset_dict['mos'] = dataset_dict['mos'] + df['dmos'].tolist()
            dataset_dict['level'] = dataset_dict['level'] + df['level'].tolist()


        dataset = Kadid10k(dataset_dict, dataset_path)

    elif dataset_name == 'kadis-cls':
        dataset = KadisFolder(dataset_path, None, None, None)
    elif dataset_name == 'kadid-F':
        dataset = KadidFolder(dataset_path, trainsz, patch_num, sel)
    elif dataset_name == 'koniq':
        dataset = Koniq_10kFolder(dataset_path, trainsz, patch_num)
    elif dataset_name == 'livec':
        dataset = LIVEChallengeFolder(dataset_path, trainsz, patch_num)
    elif dataset_name == 'csiq':
        dataset = CSIQFolder(dataset_path, trainsz, patch_num, sel)  # root, index, transform, patch_num
    elif dataset_name == 'live':
        dataset = LIVEFolder(dataset_path, trainsz, patch_num, sel)  # root, index, transform, patch_num
    # print('Data Loaded!')
    return torch.utils.data.DataLoader(
        dataset, batch_size=bs, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last, pin_memory=True)


# if __name__ == '__main__':
#     t_loader = make_dataloader(
#         dataset_name='Kadid',
#         dataset_path='/data2/Dataset/kadid10k/images',
#         csv_path='/data2/Dataset/kadid10k/25types_csv_list',
#         task_list=[1,2],
#         level_list=[1,5],
#         bs=5,
#         shuffle=True,
#         num_workers=1,
#         drop_last=True,
#         mode='all',
#         trainsz=None)
#
#     it = iter(t_loader)
#
#     print(len(it))
#     for i, [x,y] in enumerate(it):
#         print(x.size(), y)
#         if i == 0:
#             import skimage
#             skimage.io.imsave('test.png', x[0].permute(1,2,0).numpy())
