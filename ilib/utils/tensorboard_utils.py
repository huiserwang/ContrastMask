from mmcv.runner import HOOKS
from mmcv.runner.hooks.logger import TensorboardLoggerHook




@HOOKS.register_module()
class SelfTensorboardLoggerHook(TensorboardLoggerHook):
    def __init__(self,
                 log_dir=None,
                 interval=10,
                 ignore_last=True,
                 reset_flag=False,
                 by_epoch=True,
                 contr_upper_ep=4.0, 
                 contr_start_ep=0.0, 
                 init_value=0.25):
        super(SelfTensorboardLoggerHook, self).__init__(log_dir, interval, ignore_last,
                                                    reset_flag, by_epoch)
        self.contr_start_ep = contr_start_ep
        self.contr_upper_ep = contr_upper_ep
        self.init_value = init_value

    def before_epoch(self, runner):
        runner.log_buffer.clear()  # clear logs of last epoch
        weight_for_contrast = min(max(self.init_value, (runner.epoch-self.contr_start_ep)/(self.contr_upper_ep-self.contr_start_ep)), 1.0)
        runner.model.module.roi_head.mask_head.contrastive_head.weight = weight_for_contrast #change the weight of contrastive loss for different epoch
