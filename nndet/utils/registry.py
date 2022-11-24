# Modifications licensed under:
# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
# SPDX-License-Identifier: Apache-2.0
#
# L1 loss from fvcore (https://github.com/facebookresearch/fvcore) licensed under
# SPDX-FileCopyrightText: 2019, Facebook, Inc
# SPDX-License-Identifier: Apache-2.0


import inspect
import shutil
import os
from pathlib import Path
from typing import Callable


class Registry:
    def __init__(self):
        self.mapping = {}

    def __getitem__(self, key):
        return self.mapping[key]["fn"]

    def register(self, fn: Callable):
        self._register(fn.__name__, fn, inspect.getfile(fn))
        return fn

    def _register(self, name: str, fn: Callable, path: str):
        if name in self.mapping:
            raise TypeError(f"Name {name} already in registry.")
        else:
            self.mapping[name] = {"fn": fn, "path": path}

    def get(self, name: str):
        return self.mapping[name]["fn"]

    def copy_registered(self, target: Path):
        if not target.is_dir():
            target.mkdir(parents=True)
        paths = [e["path"] for e in self.mapping.values()]
        paths = list(set(paths))
        names = [p.split('nndet')[-1] for p in paths]
        names = [n.replace(os.sep, '_').rsplit('.', 1)[0] for n in names]
        names = [f"{n[1:]}.py" for n in names]
        for name, path in zip(names, paths):
            shutil.copy(path, str(target / name))
