# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
# SPDX-License-Identifier: Apache-2.0

import time

from loguru import logger


class Timer:
    def __init__(self, msg: str = "", verbose: bool = True):
        self.verbose = verbose
        self.msg = msg
        self.tic: float = None
        self.toc: float = None
        self.dif: float = None

    def __enter__(self):
        self.tic = time.perf_counter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.toc = time.perf_counter()
        self.dif = self.toc - self.tic
        if self.verbose:
            logger.info(f"Operation '{self.msg}' took: {self.toc - self.tic} sec")
