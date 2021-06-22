## Changelog
### June 2021
- (breaking) Prefixes were added to the logged values for better tensorboard support. Prefixes are, `train_loss/`, `val_loss/` (losses) and `val/` (metrics). If the default metric monitoring is used, udpate `conf.trainer_cfg.monitor_key` from `mAP_IoU_0.10_0.50_0.05_MaxDet_100` to `val/mAP_IoU_0.10_0.50_0.05_MaxDet_100`
- Improved Logging: Tensorboard, nndet_logging, additional params b8e5c06f7df2
- SAM Prototype (No Mixed Precision) 9baa16b75a64
- Additional Augmentation Modules 8aa92556d4a4
