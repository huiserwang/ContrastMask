import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from mmcv.ops.carafe import CARAFEPack
from mmcv.runner import BaseModule, ModuleList
from mmcv.cnn import ConvModule
from mmdet.models.builder import HEADS
from mmdet.models.utils import build_linear_layer

from ...utils import get_query_keys, get_query_keys_eval, enhance_op

BYTES_PER_FLOAT = 4
# TODO: This memory limit may be too much or too little. It would be better to
# determine it based on available resources.
GPU_MEM_LIMIT = 1024**3  # 1 GB memory limit

@HEADS.register_module()
class ContrastiveHead(BaseModule):
    
    def __init__(self,
                 num_convs=4,
                 num_projectfc=2,
                 roi_feat_size=28,
                 in_channels=256,
                 conv_kernel_size=3,
                 conv_out_channels=256,
                 fc_out_channels=256,
                 upsample_cfg=dict(type='deconv', scale_factor=2),
                 conv_cfg=None,
                 norm_cfg=None,
                 fc_norm_cfg=None,
                 projector_cfg=dict(type='Linear'),
                 thred_u=0.1,
                 scale_u=1.0,
                 percent=0.3,
                 init_cfg=None):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
                                 'behavior, init_cfg is not allowed to be set'

        super(ContrastiveHead, self).__init__(init_cfg)
        self.upsample_cfg = upsample_cfg.copy()
        if self.upsample_cfg['type'] not in [
                None, 'deconv', 'nearest', 'bilinear', 'carafe'
        ]:
            raise ValueError(
                f'Invalid upsample method {self.upsample_cfg["type"]}, '
                'accepted methods are "deconv", "nearest", "bilinear", '
                '"carafe"')
        self.num_convs = num_convs
        self.num_projectfc = num_projectfc
        # WARN: roi_feat_size is reserved and not used
        self.roi_feat_size = _pair(roi_feat_size)
        self.in_channels = in_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.upsample_method = self.upsample_cfg.get('type')
        self.scale_factor = self.upsample_cfg.pop('scale_factor', None)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fc_norm_cfg = fc_norm_cfg
        self.projector_cfg = projector_cfg
        self.fp16_enabled = False
        self.weight=0.0  # init variable, this will be rewrite in different epoch
        self.thred_u = thred_u
        self.scale_u = scale_u
        self.percent = percent

        #build encoder module
        self.encoder = ModuleList()
        for i in range(self.num_convs):
            in_channels = (
                self.in_channels if i == 0 else self.conv_out_channels)
            padding = (self.conv_kernel_size - 1) // 2
            self.encoder.append(
                ConvModule(
                    in_channels,
                    self.conv_out_channels,
                    self.conv_kernel_size,
                    padding=padding,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg))
            last_layer_dim = self.conv_out_channels

        #build projecter module
        self.projector = ModuleList()
        for j in range(self.num_projectfc-1):
            fc_in_channels = (
                last_layer_dim if i == 0 else self.fc_out_channels)
            self.projector.append(
                ConvModule(
                    fc_in_channels,
                    self.fc_out_channels,
                    1,
                    padding=0,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.fc_norm_cfg,)
            )
            last_layer_dim = self.fc_out_channels
        self.projector.append(
            build_linear_layer(
                self.projector_cfg,
                in_features= last_layer_dim,
                out_features = self.fc_out_channels)
        )

    def init_weights(self):
        super(ContrastiveHead, self).init_weights()

    def forward(self, x, cams, edges, masks, is_novel):
        sample_sets = dict()

        # 1. get query and keys
        if masks is not None: #training phase
            sample_results = get_query_keys(cams, edges, masks, is_novel=is_novel, thred_u=self.thred_u, scale_u=self.scale_u, percent=self.percent)
            keeps_ = sample_results['keeps']
            keeps = keeps_.reshape(-1,1,1)
            keeps = keeps.expand(keeps.shape[0], x.shape[2], x.shape[2])  # points of a reserved porposal are assigned 'keep'
            keeps_all = keeps.reshape(-1)
        else:  #evaluation phase
            sample_results = get_query_keys_eval(cams)

        # 2. forward
        for conv in self.encoder:
            x = conv(x)
        x_pro = self.projector[0](x)
        for i in range(1, len(self.projector)-1):
            x_pro = self.projector[i](x_pro)
        n, c, h, w = x_pro.shape
        x_pro = x_pro.permute(0,2,3,1).reshape(-1, c)  #n,c,h,w -> n,h,w,c -> (nhw),c
        x_pro = self.projector[-1](x_pro)              #(nhw),c
        x_enhance = enhance_op(x)

        # 3. get vectors for queries and keys so that we can calculate contrastive loss
        if masks is not None:
            query_pos_num = sample_results['query_pos_sets'].to(device=x_pro.device, dtype=x_pro.dtype).sum(dim=[1,2])
            query_neg_num = sample_results['query_neg_sets'].to(device=x_pro.device, dtype=x_pro.dtype).sum(dim=[1,2])
            assert (0. not in query_pos_num) and (0. not in query_neg_num), f"query should NOT be 0!!!! <-- contrastive_head.py"
            #keys
            sample_easy_pos = x_pro[keeps_all][sample_results['easy_positive_sets_N'].reshape(-1), :]   # *, 256
            sample_easy_neg = x_pro[keeps_all][sample_results['easy_negative_sets_N'].reshape(-1), :]   # *, 256
            sample_hard_pos = x_pro[keeps_all][sample_results['hard_positive_sets_N'].reshape(-1), :]   # *, 256
            sample_hard_neg = x_pro[keeps_all][sample_results['hard_negative_sets_N'].reshape(-1), :]   # *, 256
            #queries
            query_pos = (x_pro.reshape(-1, 28, 28, 256) * sample_results['query_pos_sets'].to(device=x_pro.device).unsqueeze(3)).sum(dim=[1,2]) / query_pos_num.unsqueeze(1) # foreground query for each proposal
            query_neg = (x_pro.reshape(-1, 28, 28, 256) * sample_results['query_neg_sets'].to(device=x_pro.device).unsqueeze(3)).sum(dim=[1,2]) / query_neg_num.unsqueeze(1)

            # sample sets are used to calculate loss
            sample_sets['keeps_proposal'] = keeps_
            sample_sets['query_pos'] = query_pos[keeps_].unsqueeze(1)
            sample_sets['query_neg'] = query_neg[keeps_].unsqueeze(1)
            sample_sets['num_per_type'] = sample_results['num_per_type']
            sample_sets['sample_easy_pos'] = sample_easy_pos
            sample_sets['sample_easy_neg'] = sample_easy_neg
            sample_sets['sample_hard_pos'] = sample_hard_pos
            sample_sets['sample_hard_neg'] = sample_hard_neg

        return x_enhance, sample_sets 
        
    def INFOloss(self, query, pos_sets, neg_sets, tem):
        ''' Dense INFOloss (pixel-wise)
        example:
            query: 5x1x256  5 is the number of proposals for a batch
            pos_sets: 135x256
            neg_sets: 170x256
        '''
        N = pos_sets.shape[0]
        N = max(1, N)

        query = torch.mean(query, dim=0).unsqueeze(0) # mean op on all proposal, sharing-query
        pos_sets = pos_sets.unsqueeze(0) - query
        neg_sets = neg_sets.unsqueeze(0) - query
        Q_pos = F.cosine_similarity(query, pos_sets, dim=2)   #[1, 135]
        Q_neg = F.cosine_similarity(query, neg_sets, dim=2)   #[1, 170]
        Q_neg_exp_sum = torch.sum(torch.exp(Q_neg/tem), dim=1)      #[1]
        single_in_log = torch.exp(Q_pos/tem) / (torch.exp(Q_pos) + Q_neg_exp_sum.unsqueeze(1))   #[1, 135]
        batch_log     = torch.sum( -1 * torch.log(single_in_log), dim=1) / N  #[1]

        return batch_log


    def loss(self, easy_pos=None, easy_neg=None, hard_pos=None, hard_neg=None, query_pos=None, query_neg=None, t_easy=0.3, t_hard=0.7):
        """
            easy_pos: [B, N, 256]
            easy_neg: [B, N, 256]
            hard_pos: [B, N, 256]
            hard_neg: [B, N, 256]
            pos_center: [B, 1, 256]
            query_pos: [B, 256]
            query_neg: [B, 256]
        """
        alpha=1.0

        loss_Qpos_easy = self.INFOloss(query_pos, easy_pos, easy_neg, t_easy)
        loss_Qpos_hard = self.INFOloss(query_pos, hard_pos, hard_neg, t_hard)
        loss_Qneg_easy = self.INFOloss(query_neg, easy_neg, easy_pos, t_easy)
        loss_Qneg_hard = self.INFOloss(query_neg, hard_neg, hard_pos, t_hard)
        loss_contrast = torch.mean(loss_Qpos_easy + loss_Qpos_hard + alpha*loss_Qneg_easy + alpha*loss_Qneg_hard)

        # check NaN
        if True in torch.isnan(loss_contrast):
            print('NaN occurs in contrastive_head loss')

        return loss_contrast * self.weight * 0.1

if __name__ == '__main__':
    con_head = ContrastiveHead()
    x = torch.randn((7, 256, 28, 28))
    y = con_head(x, None, None, None)
    
        