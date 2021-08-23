## Changelog
### August 2021
- Add `ComposePretty` to `nndet.io.augmentation.bg_aug` which enable pretty printing of the transformation pipeline
- Modular and Configurable Augmentation Pipeline
- C012 Model Tag

### June 2021
- nest MLFlow runs with same identifier 16746a86cec6787b361837e780fc0c443fb537ae
- (breaking) Prefixes were added to the logged values for better tensorboard support. Prefixes are, `train_loss/`, `val_loss/` (losses) and `val/` (metrics). If the default metric monitoring is used, udpate `conf.trainer_cfg.monitor_key` from `mAP_IoU_0.10_0.50_0.05_MaxDet_100` to `val/mAP_IoU_0.10_0.50_0.05_MaxDet_100` c4dcd5ff5b8d6ac52e21ecc0fd3dc4a4d1a301e0
- Improved Logging: Tensorboard, nndet_logging, additional params b8e5c06f7df2
- SAM Prototype (No Mixed Precision) 9baa16b75a64
- Additional Augmentation Modules 8aa92556d4a4
