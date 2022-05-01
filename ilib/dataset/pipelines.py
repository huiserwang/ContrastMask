import os.path as osp

import mmcv
import numpy as np
import pycocotools.mask as maskUtils

from mmdet.datasets.builder import PIPELINES
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.pipelines import LoadAnnotations
from mmdet.datasets.pipelines import DefaultFormatBundle
from mmdet.datasets.pipelines import to_tensor

@PIPELINES.register_module()
class LoadPSAnnotations(LoadAnnotations):
    def __init__(self,
                 with_bbox=True,
                 with_label=True,
                 with_mask=False,
                 with_seg=False,
                 poly2mask=True,
                 file_client_args=dict(backend='disk')):
        super(LoadPSAnnotations, self).__init__(
                 with_bbox=with_bbox,
                 with_label=with_label,
                 with_mask=with_mask,
                 with_seg=with_seg,
                 poly2mask=poly2mask,
                 file_client_args=file_client_args)


    def _load_is_novel(self, results):
        results['gt_is_novel'] = results['ann_info']['is_novel'].copy()
        return results

    def _load_bboxes_ids(self, results):
        results['gt_bboxes_ids'] = results['ann_info']['bboxes_ids'].copy()
        return results

    def _load_labels_names(self, results):
        results['gt_labels_names'] = results['ann_info']['labels_names'].copy()
        return results

    def __call__(self, results):
        if self.with_bbox:
            results = self._load_bboxes(results)
            if results is None:
                return None
            results = self._load_bboxes_ids(results)
        if self.with_label:
            results = self._load_labels(results)
            results = self._load_labels_names(results)
        if self.with_mask:
            results = self._load_masks(results)
            results = self._load_is_novel(results)
        if self.with_seg:
            results = self._load_semantic_seg(results)
        return results


@PIPELINES.register_module()
class DefaultFormatBundlePS(DefaultFormatBundle):
    """Default formatting bundle for partially supervised learning.

    It simplifies the pipeline of formatting common fields, including "img",
    "proposals", "gt_bboxes", "gt_bboxes_ids", "gt_labels", "gt_masks" and "gt_semantic_seg" as well as "gt_ps_masks_flag".
    These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ids: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    - gt_ps_masks_flag: (1)to tensor, (2)to DataContainer
    - gt_masks: (1)to tensor, (2)to DataContainer (cpu_only=True)
    - gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor, \
                       (3)to DataContainer (stack=True)
    """
    def __init__(self, collect_keys=['proposals', 
                                     'gt_bboxes', 
                                     'gt_bboxes_ids', 
                                     'gt_bboxes_ignore', 
                                     'gt_labels', 
                                     'gt_is_novel']):
        self.collect_keys = collect_keys

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with \
                default bundle.
        """

        if 'img' in results:
            img = results['img']
            # add default meta keys
            results = self._add_default_meta_keys(results)
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            results['img'] = DC(to_tensor(img), stack=True)
        for key in self.collect_keys: #collect_keys can be set from config
            if key not in results:
                continue
            results[key] = DC(to_tensor(results[key]))
        if 'gt_masks' in results:
            results['gt_masks'] = DC(results['gt_masks'], cpu_only=True)
        if 'gt_semantic_seg' in results:
            results['gt_semantic_seg'] = DC(
                to_tensor(results['gt_semantic_seg'][None, ...]), stack=True)
        return results