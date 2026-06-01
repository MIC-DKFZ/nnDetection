# SPDX-FileCopyrightText: 2020-2026 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from typing import Hashable, Mapping, Sequence

from nndet.io.transforms.base import AbstractTransform


class AddProps2Data(AbstractTransform):
    def __init__(self, props_key: str, key_mapping: Mapping[str, str], **kwargs):
        """
        Move properties from property dict to data dict

        Args:
            props_key: key where properties and `map_key` key is located;
            key_mapping: maps properties(key) to new keys in data dict(item)
        """
        super().__init__(grad=False, **kwargs)
        self.key_mapping = key_mapping
        self.props_key = props_key

    def forward(self, **data) -> dict:
        """
        Move keys from properties to data

        Args:
            **data: batch dict

        Returns:
            dict: updated batch
        """
        props = data[self.props_key]
        for source, target in self.key_mapping.items():
            data[target] = [p[source] for p in props]
        return data


class NoOp(AbstractTransform):
    def __init__(self, grad: bool = False):
        """
        Forward input without change

        Args:
            grad: propagate gradient through transformation
        """
        super().__init__(grad=grad)

    def forward(self, **data) -> dict:
        """
        NoOp
        """
        return data

    def invert(self, **data) -> dict:
        """
        NoOp
        """
        return data


class FilterKeys(AbstractTransform):
    def __init__(self, keys: Sequence[Hashable]):
        super().__init__(grad=False)
        self.keys = keys

    def forward(self, **data) -> dict:
        return {k: data[k] for k in self.keys}
