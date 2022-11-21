from nndet.nn.backbone.blueprints.conv import ConvBackbone
from nndet.nn.heads.detr import ConditionalDETRCEHead, ConditionalDETRHead
from nndet.nn.layers.conv import ConvInstanceRelu
from nndet.nn.transformer.conditional_transformer import ConditionalTransformer
from nndet.ptmodule import MODULE_REGISTRY
from nndet.ptmodule.detr import DETRModule


@MODULE_REGISTRY.register
class CDETR(DETRModule):
    backbone_cls = ConvBackbone
    backbone_conv_cls = ConvInstanceRelu

    transformer_cls = ConditionalTransformer
    head_cls = ConditionalDETRHead


@MODULE_REGISTRY.register
class CDETRCE(DETRModule):
    backbone_cls = ConvBackbone
    backbone_conv_cls = ConvInstanceRelu

    transformer_cls = ConditionalTransformer
    head_cls = ConditionalDETRCEHead
