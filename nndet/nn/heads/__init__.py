# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from nndet.nn.heads.abstract import BaseHead, ClassifierType, RegressorType
from nndet.nn.heads.classifier import DenseClassifier, DenseClassifierType
from nndet.nn.heads.comb import AnchorHeadType, RoIHeadType
from nndet.nn.heads.regressor import DenseRegressor, DenseRegressorType
from nndet.nn.heads.segmenter import Segmenter, SegmenterType
