import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.ops.carafe import CARAFEPack
from mmcv.runner import auto_fp16, force_fp32

from mmdet.models.roi_heads.mask_heads.fcn_mask_head import FCNMaskHead
from mmdet.models.builder import HEADS, build_head
from mmcv.cnn import build_conv_layer


@HEADS.register_module()
class FCNMaskCamHead(FCNMaskHead):

    def __init__(self,
                 num_convs=4,
                 roi_feat_size=14,
                 in_channels=256,
                 conv_kernel_size=3,
                 conv_out_channels=256,
                 num_classes=80,
                 class_agnostic=False,
                 upsample_cfg=dict(type='deconv', scale_factor=2),
                 conv_cfg=None,
                 norm_cfg=None,
                 predictor_cfg=dict(type='Conv'),
                 loss_mask=dict(
                     type='CrossEntropyLoss', use_mask=True, loss_weight=1.0),
                 init_cfg=None,
                 contrastive_head=None,
                 contrastive_enable=False):
        super(FCNMaskCamHead, self).__init__(num_convs, roi_feat_size, in_channels, conv_kernel_size,
                                            conv_out_channels, num_classes, class_agnostic, upsample_cfg,
                                            conv_cfg, norm_cfg, predictor_cfg, loss_mask, init_cfg)
        self.contrastive_enable = contrastive_enable
        
        if contrastive_head is not None:
            self.contrastive_head = build_head(contrastive_head)

        self.convs = nn.Sequential(*list(self.convs))
        self.fuseconv = build_conv_layer(predictor_cfg, 256, 256, 1)

    def init_weights(self):
        super(FCNMaskHead, self).init_weights()
        for m in [self.upsample,
                  self.conv_logits,
                  self.fuseconv
                  ]:
            if m is None:
                continue
            elif isinstance(m, CARAFEPack):
                m.init_weights()
            else:
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    @auto_fp16()
    def forward(self, x, contrastive_feats=None, bbox_cam=None, edges=None, masks=None, is_novel=None):
        if contrastive_feats is not None:
            x_contrastive, sample_sets = self.contrastive_head(contrastive_feats, 
                                                            F.interpolate(bbox_cam.detach(), size=(28,28), mode='bilinear', align_corners=False), 
                                                            edges, 
                                                            masks,
                                                            is_novel)
            
            x_contrastive_down =  F.interpolate(x_contrastive, size=(14,14), mode='bilinear', align_corners=False)
            x_contrastive_down = self.fuseconv(x_contrastive_down)
            if bbox_cam is not None:
                bbox_cam = normalize_batch(bbox_cam)
                #x = torch.cat((x, x_contrastive_down), dim=1)
                x = x + x_contrastive_down
                mask_x = x + bbox_cam 
   
        mask_pred = self.convs(mask_x)
        
        if self.upsample is not None:
            mask_pred = self.upsample(mask_pred)
            if self.upsample_method == 'deconv':
                mask_pred = self.relu(mask_pred)
        mask_pred = self.conv_logits(mask_pred)
        return mask_pred, sample_sets

    @force_fp32(apply_to=('mask_pred', ))
    def loss(self, mask_pred, mask_targets, labels):
        loss = dict()
        if mask_pred.size(0) == 0:
            loss_mask = mask_pred.sum()
        else:
            if self.class_agnostic:
                loss_mask = self.loss_mask(mask_pred, mask_targets,
                                           torch.zeros_like(labels))
            else:
                loss_mask = self.loss_mask(mask_pred, mask_targets, labels)
        loss['loss_mask'] = loss_mask

        return loss

def normalize_batch(cams_batch):
    """
    Classic min-max normalization
    Ref: https://github.com/dbtmpl/OPMask
    """
    bs = cams_batch.size(0)
    cams_batch = cams_batch + 1e-4
    cam_mins = getattr(cams_batch.view(bs, -1).min(1), 'values').view(bs, 1, 1, 1)
    cam_maxs = getattr(cams_batch.view(bs, -1).max(1), 'values').view(bs, 1, 1, 1)
    return (cams_batch - cam_mins) / (cam_maxs - cam_mins)
