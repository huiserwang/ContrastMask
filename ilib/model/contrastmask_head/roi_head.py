import torch
import mmcv
import numpy as np

import torch.nn.functional as F
from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler, mask_target
from mmdet.models.roi_heads.standard_roi_head import StandardRoIHead
from mmdet.models.builder import HEADS, build_roi_extractor

from ...utils import seg2edge
from warnings import warn


@HEADS.register_module()
class StandardRoIHeadPS(StandardRoIHead):
    """Simplest base roi head including one bbox head and one mask head."""
    def __init__(self,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 contrastive_roi_extractor=None,
                 shared_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(StandardRoIHeadPS, self).__init__(
                 bbox_roi_extractor=bbox_roi_extractor,
                 bbox_head=bbox_head,
                 mask_roi_extractor=mask_roi_extractor,
                 mask_head=mask_head,
                 shared_head=shared_head,
                 train_cfg=train_cfg,
                 test_cfg=test_cfg,
                 pretrained=pretrained,
                 init_cfg=init_cfg)

        if contrastive_roi_extractor is not None:
            self.contrastive_roi_extractor = build_roi_extractor(contrastive_roi_extractor)


    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): [5, batch, 256, H, W]
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                [batch, 10_item], each item is a type of info.
                img_metas[0]['ann_info'] has all of values

            proposals (list[Tensors]): 
            gt_bboxes (list[Tensor]): [batch, n, 4]
            gt_labels (list[Tensor]): [batch, n]
            gt_is_novel (list[Tensor]): [batch, n]
            gt_bboxes_ignore (None | list[Tensor]): None
            gt_masks (None | Tensor) : [batch, n(Bitmap)]

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    img_metas)

            losses.update(bbox_results['loss_bbox'])
            #check NaN in loss_bbox
            if True in torch.isnan(bbox_results['loss_bbox']['loss_bbox']):
                print('NaN occurs in roi_head->bbox_head->loss_bbox')
            if True in torch.isnan(bbox_results['loss_bbox']['loss_cls']):
                print('NaN occurs in roi_head->bbox_head->loss_cls')

        # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(x, sampling_results,
                                                    bbox_results['bbox_feats'], bbox_results['bbox_cam'],
                                                    gt_masks, img_metas, **kwargs)
        
            losses.update(mask_results['loss_mask'])

        return losses

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)

        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets)

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def _bbox_forward(self, x, rois):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred, bbox_cam = self.bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_cam=bbox_cam, bbox_feats=bbox_feats)
        return bbox_results

    def _mask_forward_train(self, x, sampling_results, bbox_feats, bbox_cam, gt_masks,
                            img_metas, **kwargs):
        """Run forward function and calculate loss for mask head in
        training."""

        """
            sampling_results: list[SamplingResult]->len=batch
        """
        """
            pos_roi: [batch1*n1+batch2*n2, 5] where batch1+batch2=2
            mask_results: dict, keys='mask_pred', 'edge_pred', 'mask_feat'
        """

        #There are some difference: get mask_targets firstly, so that edge_targets can be used in func:_mask_forward
        mask_targets = self.mask_head.get_targets(sampling_results, gt_masks, self.train_cfg)  #mask_targets: [n1+n2, 28, 28]
        edge_targets = seg2edge(mask_targets)  

        # here we ONLY compute mask loss on novel instances
        is_novel = []
        gt_proposal_ = []
        my_pos_labels = []
        for i, meta in enumerate(img_metas):
            # get pos_assigned_gt_ids for each sample. It is used to assign some property for non-GT pos proposals
            pos_assigned_gt_inds = sampling_results[i].pos_assigned_gt_inds
            pos_is_gt_ = sampling_results[i].pos_is_gt
            is_novel_ = torch.tensor(meta["ann_info"]["is_novel"])
            is_novel_ = is_novel_[pos_assigned_gt_inds]
            is_novel.append(is_novel_)
            gt_proposal_.append(pos_is_gt_)
            my_pos_labels.append(torch.tensor(meta["ann_info"]["labels"]).to(pos_assigned_gt_inds)[pos_assigned_gt_inds])
        #verification
        my_pos_labels = torch.cat(my_pos_labels)
        assert torch.all(my_pos_labels == torch.cat([res.pos_gt_labels for res in sampling_results]))

        #concat aforementioned properties of different samples into one tensor
        is_novel = torch.cat(is_novel, dim=0).to(bool)
        is_base = torch.logical_not(is_novel)
        gt_proposal = torch.cat(gt_proposal_, dim=0).to(is_base)
        valid_masks = torch.logical_and(is_base, gt_proposal)
            
        if not self.share_roi_extractor:
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
            pos_rois_indexs_ = [list(range(i * 512, len(n.pos_is_gt) + i*512)) for i,n in enumerate(sampling_results)]
            pos_rois_indexs = torch.tensor(np.concatenate(pos_rois_indexs_, axis=0))
            pos_labels_for_cam = torch.cat([res.pos_gt_labels for res in sampling_results], dim=0)
            mask_results = self._mask_forward(x, pos_rois, bbox_cam=bbox_cam[pos_rois_indexs], pos_labels_for_cam=pos_labels_for_cam, edges=edge_targets, masks=mask_targets, is_novel=is_novel)
        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(
                    torch.ones(
                        res.pos_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
                pos_inds.append(
                    torch.zeros(
                        res.neg_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
            pos_inds = torch.cat(pos_inds)

            mask_results = self._mask_forward(
                x, pos_inds=pos_inds, bbox_feats=bbox_feats, bbox_cam=bbox_cam)

        #select final pred and targets for calculating loss
        mask_pred = mask_results['mask_pred']
        contrastive_sets = mask_results['contrastive_sets']
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])

        #calculate losses through the built-in loss func in mask_head.
        #here, only base(seen) categories are considered when calculating loss_mask.
        loss_mask = self.mask_head.loss(mask_pred[valid_masks],
                                        mask_targets[valid_masks],
                                        pos_labels[valid_masks])

        #calculate contrastive loss for base and novel
        if contrastive_sets is not None:
            loss_contrastive = self.mask_head.contrastive_head.loss(contrastive_sets['sample_easy_pos'],
                                                                    contrastive_sets['sample_easy_neg'],
                                                                    contrastive_sets['sample_hard_pos'],
                                                                    contrastive_sets['sample_hard_neg'],
                                                                    contrastive_sets['query_pos'],
                                                                    contrastive_sets['query_neg'],
                                                                    t_easy=0.3,
                                                                    t_hard=0.7)
            loss_mask['loss_contrastive'] = loss_contrastive
            loss_mask['weight_for_cl'] = torch.tensor(self.mask_head.contrastive_head.weight).cuda()
            loss_mask['pred_num_all'] = torch.tensor(float(mask_pred.shape[0])).cuda()
            loss_mask['pred_num_base'] = torch.tensor(float(mask_pred[valid_masks].shape[0])).cuda()

        # update mask_results and return it
        mask_results.update(loss_mask=loss_mask, mask_targets=mask_targets)
        return mask_results

    def _mask_forward(self, x, rois=None, pos_inds=None, bbox_feats=None, bbox_cam=None, pos_labels_for_cam=None, edges=None, masks=None, is_novel=None):
        """Mask head forward function used in both training and testing."""
        assert ((rois is not None) ^
                (pos_inds is not None and bbox_feats is not None))
        if rois is not None:
            #get roi_feats for mask branch, termed as mask_feats
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], rois)
            #get roi_feats for contrastive branch, termed as contrastive_feats
            if self.mask_head.contrastive_enable:
                contrastive_feats = self.contrastive_roi_extractor(
                    x[:self.mask_roi_extractor.num_inputs], rois)
            else:
                contrastive_feats = None
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
        else:
            assert bbox_feats is not None
            mask_feats = bbox_feats[pos_inds]
        #select ONE cam according to gt_labels or predicted results when training or testing
        if pos_labels_for_cam is not None:
            batch_num_slice = torch.arange(len(pos_labels_for_cam))
            bbox_cam_single = bbox_cam[batch_num_slice, pos_labels_for_cam,:,:]
            bbox_cam_single = torch.unsqueeze(bbox_cam_single, dim=1)
        else:
            bbox_cam_single = bbox_cam
        #upsample the cam from 7x7 to 14x14, so that it can be added to mask_feats
        bbox_cam_single_upsample = F.interpolate(bbox_cam_single, size=(14,14), mode='bilinear', align_corners=False)

        mask_pred, contrastive_sets = self.mask_head(mask_feats, 
                                                    contrastive_feats=contrastive_feats.detach(), 
                                                    bbox_cam=bbox_cam_single_upsample,
                                                    edges=edges,
                                                    masks=masks,
                                                    is_novel=is_novel)
        mask_results = dict(mask_pred=mask_pred, mask_feats=mask_feats, contrastive_sets=contrastive_sets)

        return mask_results

    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False,
                    get_gt_mask=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels, det_bboxes_cams = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]

        if get_gt_mask:
            device = det_bboxes[0].device
            gt_bboxes_ignore = [None for _ in img_metas]
            gt_bboxes = [torch.tensor(meta['gt_bboxes']).to(device) for meta in img_metas]
            gt_labels = [torch.tensor(meta['gt_labels']).to(device) for meta in img_metas]
            gt_masks = [meta['gt_masks'] for meta in img_metas]

            assigner_cfg = dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.1,
                    neg_iou_thr=0.1,
                    min_pos_iou=0.1,
                    match_low_quality=False,
                    ignore_iof_thr=-1)
            sampler_cfg = dict(
                type='RandomSampler',
                num=100,
                pos_fraction=1.0,
                neg_pos_ub=-1,
                add_gt_as_proposals=False)
            bbox_assigner = build_assigner(assigner_cfg)
            bbox_sampler = build_sampler(sampler_cfg)
            sampling_results = []
            for i, meta in enumerate(img_metas):
                assign_result = bbox_assigner.assign(
                    det_bboxes[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i]
                )
                bbox_sampler.num = det_bboxes[0].shape[0]
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    det_bboxes[i],
                    gt_bboxes[i],
                    gt_labels[i]
                )
                sampling_results.append(sampling_result)

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, det_bboxes_cams, rescale=rescale)
            return list(zip(bbox_results, segm_results))

    def simple_test_bboxes(self,
                           x,
                           img_metas,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False):
        """Test only det bboxes without augmentation.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            proposals (List[Tensor]): Region proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            tuple[list[Tensor], list[Tensor]]: The first list contains
                the boxes of the corresponding image in a batch, each
                tensor has the shape (num_boxes, 5) and last dimension
                5 represent (tl_x, tl_y, br_x, br_y, score). Each Tensor
                in the second list is the labels with shape (num_boxes, ).
                The length of both lists should be equal to batch_size.
        """

        rois = bbox2roi(proposals)
        bbox_results = self._bbox_forward(x, rois)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # split batch bbox prediction back to each image
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']
        bbox_cam  = bbox_results['bbox_cam']
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)

        # some detector with_reg is False, bbox_pred will be None
        if bbox_pred is not None:
            # TODO move this to a sabl_roi_head
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
                bbox_cam  = bbox_cam.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head.bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
                #TODO
                #bbox_cam
        else:
            bbox_pred = (None, ) * len(proposals)
            bbox_cam = (None, ) * len(proposals)

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        det_bboxes_cams = []
        for i in range(len(proposals)):
            det_bbox, det_label, keep_inds_ = self.bbox_head.get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
            #detele the background dim and reshape into [num_proposal x num_class, 7, 7], then slice
            bbox_cam_i = bbox_cam[i][:,:-1,:,:]
            bbox_cam_i = bbox_cam_i.reshape(-1, 7, 7)
            det_bboxes_cams.append(bbox_cam_i[keep_inds_].unsqueeze(1))

        return det_bboxes, det_labels, det_bboxes_cams

    def simple_test_mask(self,
                         x,
                         img_metas,
                         det_bboxes,
                         det_labels,
                         det_bboxes_cams,
                         rescale=False):
        """Simple test for mask head without augmentation."""
        # image shapes of images in the batch
        ori_shapes = tuple(meta['ori_shape'] for meta in img_metas) #original input image shape
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        if isinstance(scale_factors[0], float):
            logger.warning(
                'Scale factor in img_metas should be a '
                'ndarray with shape (4,) '
                'arrange as (factor_w, factor_h, factor_w, factor_h), '
                'The scale_factor with float type has been deprecated. ')
            scale_factors = np.array([scale_factors] * 4, dtype=np.float32)

        num_imgs = len(det_bboxes)
        if all(det_bbox.shape[0] == 0 for det_bbox in det_bboxes):
            segm_results = [[[] for _ in range(self.mask_head.num_classes)]
                            for _ in range(num_imgs)]
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            if rescale:
                scale_factors = [
                    torch.from_numpy(scale_factor).to(det_bboxes[0].device)
                    for scale_factor in scale_factors
                ]
            _bboxes = [
                det_bboxes[i][:, :4] *
                scale_factors[i] if rescale else det_bboxes[i][:, :4]
                for i in range(len(det_bboxes))
            ]
            mask_rois = bbox2roi(_bboxes)

            mask_bboxes_cams = torch.cat(det_bboxes_cams, dim=0)
            # mask_labels_pred = torch.cat(det_labels, dim=0)
            mask_results = self._mask_forward(x, mask_rois, bbox_cam=mask_bboxes_cams, pos_labels_for_cam=None) #NO need to slice
            mask_pred = mask_results['mask_pred']
            # split batch mask prediction back to each image
            num_mask_roi_per_img = [len(det_bbox) for det_bbox in det_bboxes]
            mask_preds = mask_pred.split(num_mask_roi_per_img, 0)

            # apply mask post-processing to each image individually
            segm_results = []
            for i in range(num_imgs):
                if det_bboxes[i].shape[0] == 0:
                    segm_results.append(
                        [[] for _ in range(self.mask_head.num_classes)])
                else:
                    segm_result = self.mask_head.get_seg_masks(
                        mask_preds[i], _bboxes[i], det_labels[i],
                        self.test_cfg, ori_shapes[i], scale_factors[i],
                        rescale)
                    segm_results.append(segm_result)

        return segm_results
