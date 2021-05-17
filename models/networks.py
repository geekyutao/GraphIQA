import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torchvision import models
from torchvision.models.resnet import model_urls

from .modules import ResnetFeatureExtractor, DomainLevelGragh, GCN


# DualGragh Regression
class Reg_Domain(nn.Module):
    def __init__(self, do_emb_size, eg_emb_size):
        super(Reg_Domain, self).__init__()

        self.extractor = ResnetFeatureExtractor(layer=50, pretrained=True)

        for key, p in self.named_parameters():
            # print(key)
            p.requires_grad = False

        self.domainlevelgraph = DomainLevelGragh(2048, do_emb_size, eg_emb_size)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.predictor = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1)
            )
        self.hyperpredmos = HyperPred(2048 + eg_emb_size)

        self.classifier = nn.Linear(256, 25)


        for m in self.modules():
            self.weights_init(m)


    def forward(self, x):
        '''
        x: (N, C, H, W); In Kadid-10k, (N, 3, 224, 224).
        N: batch size (i.e. number of domain graph nodes)
        '''
        x = self.extractor(x)   # (N, 2048, 7, 7)

        ins_emb, eg_emb_eg, level_pred, do_emb = self.domainlevelgraph(x)

        # regression
        mean, scale = self.hyperpredmos(torch.cat([ins_emb, eg_emb_eg], -1))
        x = self._mos_vae(mean, scale)

        # node only
        # mean, scale = self.hyperpredmos(do_emb)
        # x = self._mos_vae(mean, scale)

        # # edge only
        # mean, scale = self.hyperpredmos(eg_emb_eg)
        # x = self._mos_vae(mean, scale)

        #
        # x = self.global_pool(x).view(x.size(0), -1)
        # x = self.predictor(x)   # (N, M) -> (N, 1)

        do_code = do_emb.mean(0)
        type_pred = self.classifier(do_emb)

        return x, do_code, level_pred, type_pred


    def weights_init(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def _mos_vae(self,mean,scale):
        # (N, P)
        noise = torch.randn(mean.size()).cuda()
        mos_pred = mean + noise * scale
        return mos_pred

class HyperPred(nn.Module):
    def __init__(self, in_dim, out_dim=2):
        super(HyperPred, self).__init__()

        self.fc = nn.Sequential(nn.Linear(in_dim, in_dim // 2, bias=True),
                                  nn.ReLU(),
                                  nn.Linear(in_dim // 2, out_dim, bias=True))
        # for m in self.modules():
        #     self.weights_init(m)

    def forward(self, x):
        # input: (N, K)
        # output: (N, 2)
        x = self.fc(x)
        mean, scale = x.split(1, dim=1)  # (N, 1) * 2
        return mean, scale

    # def weights_init(self, m):
    #     if isinstance(m, nn.Linear) or isinstance(m, nn.Bilinear):
    #         torch.nn.init.xavier_uniform_(m.weight.data)
    #         if m.bias is not None:
    #             m.bias.data.fill_(0.0)