from dynamic_network_architectures.building_blocks.eva import Eva
from dynamic_network_architectures.building_blocks.patch_encode_decode import LayerNormNd, PatchEmbed
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from einops import rearrange
from loguru import logger
from torch import nn

from nndet.nn.backbone.abstract_primus import PrimusAbstract, WrapperPrimusAbstractBackbone
from nndet.nn.backbone.dynamic_network.vit_embed_decode import PatchDecode


class Primus_backbone(PrimusAbstract):
    def __init__(self,
                 config):
        """
        Primus attention-based encode.
        """
        input_channels = config['plan_arch']['in_channels']
        output_channels= config['plan_arch']['seg_classes']
        input_shape = config['backbone_cfg']['patch_size']
        embed_dim = config['backbone_cfg']['embed_dim']
        patch_embed_size = config['backbone_cfg']['patch_embed_size']
        eva_depth = config['backbone_cfg']['encoder_eva_depth']
        eva_numheads = config['backbone_cfg']['encoder_eva_numheads']
        drop_path_rate = config['backbone_cfg']['drop_path_rate']
        init_values = config['backbone_cfg']['init_values'][0] if type(config['backbone_cfg']['init_values'])==list else config['backbone_cfg']['init_values']
        scale_attn_inner = config['backbone_cfg']['scale_attn_inner']

        assert input_shape is not None
        assert len(input_shape) == 3, "Currently only 3D is supported"
        assert all([j % i == 0 for i, j in zip(patch_embed_size, input_shape)])

        super().__init__()
        self.config=config
        self.embed_dim = embed_dim
        self.input_shape = input_shape
        self.patch_embed_size = patch_embed_size
        self.input_channels = input_channels
        # Patch embedding for encoder
        self.down_projection = PatchEmbed(patch_embed_size, input_channels, embed_dim)

        # Encoder using EVA
        self.eva = Eva(
            embed_dim=embed_dim,
            depth=eva_depth,
            num_heads=eva_numheads,
            ref_feat_shape=tuple([i // ds for i, ds in zip(input_shape, patch_embed_size)]),
            drop_path_rate=drop_path_rate,
            init_values=init_values,
            scale_attn_inner=scale_attn_inner,
        )

        # Patch embedding for decoder
        self.up_projection = PatchDecode(patch_embed_size, embed_dim, output_channels,
                                         norm=LayerNormNd,
                                         activation=nn.GELU)

        self.down_projection.apply(InitWeights_He(1e-2))

    def forward(self, x):
        # Encode patches
        x = self.down_projection(x)
        B, C, W, H, D = x.shape
        x = rearrange(x, 'b c w h d -> b (w h d) c')

        # Encode using EVA (internally applies masking with patch_drop_rate)
        encoded, keep_indices = self.eva(x)

        # Project back to output shape
        decoded = rearrange(encoded, 'b (w h d) c -> b c w h d', h=H, w=W, d=D)
        decoded = self.up_projection(decoded)

        return decoded

    def get_channels(self):
        # hardcoded for now
        return self.up_projection.out_channels[:2]

class PrimusbackBoneWrapper(WrapperPrimusAbstractBackbone):
    def __init__(
        self,
        config,
    ) -> None:
        super().__init__()
        """
        Backbone with convnext blocks and conv stem
        """

        self.backbone=self.from_config_plan(config['backbone_kwargs'],config['plan_arch'])

    @classmethod
    def from_config_plan(
        cls,
        backbone_cfg: dict,
        plan_arch: dict,
    ):
        """
        Instantiate Backbone from given configs.

        Args
            backbone_cfg: backbone configuration
        """
        logger.info(f"Building:: backbone {cls.__name__}: {backbone_cfg} ")
        logger.info(f"Building:: Arch {cls.__name__} (content mainly not used): {plan_arch} ")
        # parse config and plan

        #for now i fix everything
        config={'backbone_cfg': backbone_cfg, 'plan_arch': plan_arch}
        backbone = Primus_backbone(config=config,)

        return backbone
