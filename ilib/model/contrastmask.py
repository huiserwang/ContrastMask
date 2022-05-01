from mmdet.models.builder import DETECTORS
from mmdet.models.detectors import TwoStageDetector

@DETECTORS.register_module()
class ContrastMask(TwoStageDetector):
    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None):
        super(ContrastMask, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)

    
    def train_step(self, data, optimizer, **kwargs):
        """
            The iteration step during training
        """
        # forward
        losses = self(**data, **kwargs) 

        #parse loss
        loss, log_vars = self._parse_losses(losses)

        #make output
        outputs = dict(loss=loss,
                       log_vars=log_vars,
                       num_samples=len(data['img_metas']))

        #return
        return outputs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_is_novel,
                      gt_bboxes_ids,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        # backbone forward
        '''
            img: tensor->[Batchsize, 3, H, W]
            x  : tuple(tensor)->len=5, tensor_shape:[batchsize, 256, H/s, W/s], feature maps
        '''
        x = self.extract_feat(img)

        # build dict for recording loss
        losses = dict()

        # RPN forward and loss
        '''
            proposal_list: list(tensor)->len=batchsize, tensor_shape:[1000,5], 1000 means anchors, 5 means pos and score
        '''
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels, gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)

        return losses