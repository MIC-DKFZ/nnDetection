from loguru import logger

from nndet.io.augmentation import AUGMENTATION_REGISTRY
from nndet.io.augmentation.base import ComposePretty
from nndet.io.augmentation.pipeline.modular import AugModular
from nndet.io.augmentation.torchio import TIOTransform

try:
    import torchio as tio
except ImportError:
    tio = None


@AUGMENTATION_REGISTRY.register
class MRIAugModular(AugModular):
    """
    Started out as a direct copy of `BaseInsaneAug` but can be configured
    in various ways to increase / decrease the augmentation strength
    """

    def get_training_transforms(self):
        """
        AugModular:
            - UtilTransforms
            - SpatialTransform
            - GaussianNoiseTransform
            - GaussianBlurTransform
            - BrightnessMultiplicativeTransform
            - [optional] BrightnessTransform
            - ContrastAugmentationTransform
            - [optional] SimulateLowResolutionTransform
            - [optional] GammaTransform (inverted)
            - [optional] GammaTransform
            - [optional] MirrorTransform
            - UtilTransforms
        MRI Transforms:
            - TBD
        """
        print("THIS IS A PLACEHOLDER")
        tr_transforms = [super().get_training_transforms()]

        mri_transforms = []
        mri_transforms.append(
            tio.transforms.RandomBiasField(p=1.0),
        )

        trafo = TIOTransform(
            trafo=tio.transforms.Compose(mri_transforms), data_key="data"
        )

        logger.info(f"Training transforms were extended with \n{mri_transforms}")
        tr_transforms.append(trafo)
        return ComposePretty(tr_transforms)
