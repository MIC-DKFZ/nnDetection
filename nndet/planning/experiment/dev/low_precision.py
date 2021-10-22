from typing import Dict

from nndet.planning.experiment import PLANNER_REGISTRY
from nndet.planning.experiment.v001 import D3V001
from nndet.preprocessing.preprocessor.others import PreprocessorFP16I16


@PLANNER_REGISTRY.register
class D3V001FP16I16(D3V001):
    @staticmethod
    def create_preprocessor(plan: Dict) -> PreprocessorFP16I16:
        """
        Create Preprocessor
        """
        preprocessor = PreprocessorFP16I16(
            norm_scheme_per_modality=plan["normalization_schemes"],
            use_mask_for_norm=plan["use_mask_for_norm"],
            transpose_forward=plan["transpose_forward"],
            intensity_properties=plan["dataset_properties"]["intensity_properties"],
            resample_anisotropy_threshold=plan["resample_anisotropy_threshold"],
        )
        return preprocessor
