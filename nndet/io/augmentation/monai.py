from typing import Optional

from batchgenerators.transforms.abstract_transforms import AbstractTransform

try:
    import monai

    class MonaiTransform(AbstractTransform):
        def __init__(
            self,
            trafo: monai.transforms.Transform,
            data_key: str = "data",
            label_key: Optional[str] = "target",
        ) -> None:
            """
            This is a minimal wrapper around `Monai` to apply transformations
            which don't require an additional label

            Args:
                trafo: `monai` based dict transformation. If more than one
                    transformation is used, please wrap them
                    in a compose function of `monai` first.
                data_key: key where data is located for transformation
            """
            super().__init__()
            self.trafo = trafo
            self.data_key = data_key
            self.label_key = label_key

        def __call__(self, **data_dict):
            batch_size = data_dict[self.data_key].shape[0]

            for b in range(batch_size):
                # extract current element from batch dict
                element_dict = {
                    key: data_dict[key][b]
                    for key in [self.data_key, self.label_key]
                    if key is not None
                }

                # augment
                augmented_element_dict = self.trafo(element_dict)

                # save into batch dict
                for key, item in augmented_element_dict.items():
                    data_dict[key][b] = item
            return data_dict


except ImportError:
    monai = None
