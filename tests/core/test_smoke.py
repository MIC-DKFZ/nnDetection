# import torch

# from nndet.core.rois.head import RoIModule
# from nndet.core.rois.pooler import RoIAlignNaiveAssign
# from nndet.arch.heads.classifier.roi import RoIClassifierTwoMLP
# from nndet.arch.heads.regressor.roi_single import RoIRegressorConv
# from nndet.arch.heads.comb.roi import RoIBoxHead
# from nndet.core.boxes.coder import BoxCoderND
# from nndet.arch.conv import Generator, ConvInstanceRelu, ConvGroupRelu
# from nndet.core.boxes.ops import box_iou
# from nndet.core.boxes.matcher import IoUMatcher
# from nndet.core.boxes.sampler import NegativeSampler


# def test_coder_smoke():
#     output_size = (7, 7)
#     conv = Generator(ConvInstanceRelu, 2)
    
#     coder = BoxCoderND(weights=(1.,) * (2 * 2))
#     classifier = RoIClassifierTwoMLP(
#         conv=conv,
#         in_channels = 16 * 7 * 7,
#         internal_channels=128,
#         num_classes=1
#     )
#     regressor = RoIRegressorConv(
#         conv=conv,
#         in_channels=16,
#         internal_channels=32,
#     )
#     roi_head = RoIBoxHead(
#         classifier=classifier,
#         regressor=regressor,
#         coder=coder,
#     )
#     matcher = IoUMatcher(
#             low_threshold=0.3,
#             high_threshold=0.5,
#             allow_low_quality_matches=True,
#         )
#     pooler = RoIAlignNaiveAssign(
#         output_size=output_size
#     )
#     sampler = NegativeSampler(
#         batch_size_per_image=32,
#         positive_fraction=0.5,
#     )

#     module = RoIModule(
#         box_head=roi_head, # use head without sampler
#         matcher=matcher,
#         pooler=pooler,
#         sampler=sampler, # NegativeSampler default => random balanced sampling
#         num_classes=1,
#         decoder_levels=(0, 1, 2),
#         gt_to_proposals=True,
#     )

#     images = torch.rand(1, 1, 32, 32)
#     feature_maps = [torch.rand(1, 16, 8, 8)] * 3
#     proposals = {
#         "pred_boxes": [torch.tensor([[0., 0., 16., 16.],
#                                      [0., 0., 12., 12.],
#                                      [16., 16., 32., 32.]])],
#         "pred_scores": [torch.tensor([1.0, 1.0])],
#     }
#     gt = {
#         "target_boxes": [torch.tensor([[0., 0., 16., 16.]])],
#         "target_classes": [torch.tensor([0, 0])],
#     }
#     # module._train_step_boxes(
#     #     images=images,
#     #     features=feature_maps,
#     #     proposals=proposals,
#     #     targets=gt,
#     # )
    
#     module.inference_step(
#         images=images,
#         features=feature_maps,
#         proposals=proposals,
#     )
