import os
import torch
import torch.utils.data as data

from PIL import Image
from torchvision.transforms import Compose, ToTensor, CenterCrop, RandomCrop, Normalize, Resize

class Kadid10k(data.Dataset):
    def __init__(self, dataset_dict, dataset_path):

        self.dataset_dict = dataset_dict
        self.dataset_path = dataset_path

        # self.transform = self.img_transform(384)
        self.transform = self.img_transform(224)

    def __len__(self):
        return len(self.dataset_dict['img'])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, mos)
        """

        img_path = os.path.join(self.dataset_path, self.dataset_dict['img'][index])
        # img_path = self.dataset_dict['img'][index]
        img = Image.open(img_path, mode='r').convert('RGB')
        img = self.transform(img)

        mos = self.dataset_dict['mos'][index]
        mos = torch.Tensor([mos])

        level = self.dataset_dict['level'][index]
        level = torch.Tensor([level])


        return img, mos, level


    def img_transform(self, crop_size):
        normalize = Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        return Compose([
            RandomCrop(crop_size),
            # Resize([crop_size, crop_size]),
            # Resize(crop_size),
            ToTensor(),
            normalize
            ])




# if __name__ == '__main__':
#     dataset = ImitationKadid10k(flist='trainlist.txt',
#                                 dataset_path='/data/dataset/kadid10k/images/',
#                                 transform=None,
#                                 target_transform=None,
#                                 train=True,
#                                 all=True)
#     demo, rank = dataset[3]
#     print(demo.size(), rank)
