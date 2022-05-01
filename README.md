## ContrastMask: Contrastive Learning to Segment Every Thing

[arXiv](https://arxiv.org/abs/2203.09775) | [Project page](https://blog.huiserwang.site/2022-03/Project-ContrastMask)

### News
- 2022/05/04 Code is available.
- 2022/03/02 ContrastMask is accepted to **CVPR2022**
---
### Abstract

Partially-supervised instance segmentation is a task which requests segmenting objects from novel categories via learning on limited base categories with annotated masks thus eliminating demands of heavy annotation burden. The key to addressing this task is to build an effective class-agnostic mask segmentation model. Unlike previous methods that learn such models only on base categories, in this paper, we propose a new method, named ContrastMask, which learns a mask segmentation model on both base and novel categories under a unified pixel-level contrastive learning framework. In this framework, annotated masks of base categories and pseudo masks of novel categories serve as a prior for contrastive learning, where features from the mask regions (foreground) are pulled together, and are contrasted against those from the background, and vice versa. Through this framework, feature discrimination between foreground and background is largely improved, facilitating learning of the class-agnostic mask segmentation model. Exhaustive experiments on the COCO dataset demonstrate the superiority of our method, which outperforms previous state-of-the-arts.

---

### Requirements
- cuda == 10.1
- Pytorch == 1.7.0
- MMDetection == 2.14.0
- mmcv-full == 1.3.8

---

### Training and Evaluation
> Note: we reorganize the code from the initial version to make it more readable
Our code is based on MMDetection, thus the training and evaluation pipeline are the same with that in MMDetection. Here, we give an example script to train a model with res-50 backbone under 1x schedule for voc->nonvoc setting.

``` shell
bash ./iscript/dist_train.sh myconfig/ablation/conmask_r50_fpn_3x_full_voc_nonvoc_wo_cam72.py --deterministic --seed 0 --work-dir $WORKDIR --work_id $WORKID --exp_details $CFG_DETAILS
```

If you want to try the nonvoc->voc setting, you can revise the term ``novel`` in config file from voc to nonvoc.

----

### Pretrained Models
> Note: Due to the random sampling operation for generating different types of queries and keys, the final performance of the model would have a minor perturbation of about $\pm$ 0.2 mAP.

| Backbone | setting | sche | mAP | download |
| :---- | :---- | :----: | :----: | :---- |
| ResNeXt-50 | voc -> nonvoc | 3x | 33.4 | [OneDrive](https://sjtueducn-my.sharepoint.com/:u:/g/personal/wangxuehui_sjtu_edu_cn/Eb7Oi9EwkVdEneNXHIHDyHUBzKPMxwxZbd4eJ4f2cojOfA?e=yK6Nha) (password: huiser) <br> [SJTU-Box](https://jbox.sjtu.edu.cn/l/r10gKe) (password: huiser)|
| ResNeXt-50 | nonvoc -> voc | 3x | 37.6 | [OneDrive](https://sjtueducn-my.sharepoint.com/:u:/g/personal/wangxuehui_sjtu_edu_cn/ESrME6s0ZqJPndb5XNN9SroBHRNFQSgiWGbwm89VRS7Rtg?e=kfUNtF) (password: huiser) <br> [SJTU-Box](https://jbox.sjtu.edu.cn/l/d1z6GB) (password: huiser)|
| ResNeXt-101 | voc -> nonvoc | 3x | 35.0 | [OneDrive](https://sjtueducn-my.sharepoint.com/:u:/g/personal/wangxuehui_sjtu_edu_cn/EZMw5g5mN41FjhTsztO7Ax0BDqOhfO3Shj6Nzjnnq6QYtw?e=q9Whom) (password: huiser) <br> [SJTU-Box](https://jbox.sjtu.edu.cn/l/91XHnI) (password: huiser)|
| ResNeXt-101 | nonvoc -> voc | 3x | 39.8 | [OneDrive](https://sjtueducn-my.sharepoint.com/:u:/g/personal/wangxuehui_sjtu_edu_cn/EWCIZTC51UVMvIedcaf-GKsBRQZZV2WTtTKhGhqKoAg6nw?e=Eq4MLj) (password: huiser) <br> [SJTU-Box](https://jbox.sjtu.edu.cn/l/51uUwp) (password: huiser)|

### Acknowledgement
- We thanks for the excellent [MMDetection](https://github.com/open-mmlab/mmdetection) which makes us start our work easily.
