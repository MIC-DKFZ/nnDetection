from typing import Any, Sequence

import torch


class AbstractTransform(torch.nn.Module):
    def __init__(self, grad: bool = False, **kwargs):
        """
        Args:
            grad: enable gradient computation inside transformation
        """
        super().__init__()
        self.grad = grad

    def __call__(self, *args, **kwargs) -> Any:
        """
        Call super class with correct torch context

        Args:
            *args: forwarded positional arguments
            **kwargs: forwarded keyword arguments

        Returns:
            Any: transformed data

        """
        if self.grad:
            context = torch.enable_grad()
        else:
            context = torch.no_grad()

        with context:
            return super().__call__(*args, **kwargs)


class Compose(AbstractTransform):
    def __init__(self, *transforms):
        """
        Compose multiple transforms to one
        
        Args:
            transforms: transformations to compose
        """
        super().__init__(grad=False)
        if len(transforms) == 1 and isinstance(transforms[0], Sequence):
            transforms = transforms[0]

        self.transforms = torch.nn.ModuleList(list(transforms))

    def forward(self, **batch):
        """
        Augment batch
        """
        for t in self.transforms:
            batch = t(**batch)
        return batch
