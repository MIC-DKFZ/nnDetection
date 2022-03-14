from batchgenerators.transforms.abstract_transforms import AbstractTransform

try:
    import torchio as tio
except ImportError:
    tio = None


class TIOTransform(AbstractTransform):
    def __init__(
        self,
        trafo: "tio.transforms.Transform",
        data_key: str = "data",
    ) -> None:
        """
        This is a minimal wrapper around TorchIO to apply transformations
        which don't require an additional label

        Args:
            trafo: TorchIO based transformation. If more than one
                TochIO transformation is used, please wrap them
                in a compose function of TorchIO first.
            data_key: key where data is located for transformation
        """
        super().__init__()
        if tio is None:
            raise ImportError("MonaiTransform requriees MONAI but was not found!")
        self.trafo = trafo
        self.data_key = data_key

    def __call__(self, **data_dict):
        batch_size = data_dict[self.data_key].shape[0]
        for b in range(batch_size):
            data_dict[self.data_key][b] = self.trafo(data_dict[self.data_key][b])
        return data_dict
