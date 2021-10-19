## Changelog
### nnDetection v0.1.3-dev
- Tox Testautomation aa8a35f3e0d48777aa91c86498ac222e41172483
- C014 [tag switch of padding in predictor]
- (breaking) Encoder / Decoder Names in Config are now deprecated
- (breaking) Introduced new model building names: Encoder->Backbone, Decoder->Neck
- (breaking) PTModules are now composed from mixins which defines tasks and model configuration
- `nndet_eval` won't run evluation on preprocessed data by default and now needs a flag 3c980c39f929
- New entrypoints: `nndet_prep_labels`, `nndet_print_reg` 333b98ce5a44 , 777174d7eca3

### nnDetection v0.1.2-dev
- pre-commit CI with pytest and black 606296cfc16e
- Many additional unittests
- Add `ComposePretty` to `nndet.io.augmentation.bg_aug` which enable pretty printing of the transformation pipeline 8629d74b70cf
- Modular and Configurable Augmentation Pipeline 7c6eb0391dc0
- C012 Model Tag 7c6eb0391dc0
- nest MLFlow runs with same identifier 16746a86cec6787b361837e780fc0c443fb537ae
- (breaking) Prefixes were added to the logged values for better tensorboard support. Prefixes are, `train_loss/`, `val_loss/` (losses) and `val/` (metrics). If the default metric monitoring is used, udpate `conf.trainer_cfg.monitor_key` from `mAP_IoU_0.10_0.50_0.05_MaxDet_100` to `val/mAP_IoU_0.10_0.50_0.05_MaxDet_100` c4dcd5ff5b8d6ac52e21ecc0fd3dc4a4d1a301e0
- Improved Logging: Tensorboard, nndet_logging, additional params b8e5c06f7df2
- SAM Prototype (No Mixed Precision) 9baa16b75a64
- Additional Augmentation Modules 8aa92556d4a4

### nnDetection v0.1.1-dev
- transfer learning setup

### nnDetection v0.1
release